"""Shruti (a fine-tune of Cactus Whistle, Apache-2.0) as a PyTorch inference model, for ONNX export.

A line-by-line port of needle's JAX forward pass (needle/model/whistle.py + architecture.py, pinned
commit 482361d), float weights, no activation fake-quantisation. Three exportable graphs:
- Frontend: 16 kHz samples (1, N) -> normalised log-mel features (1, T, 80). The STFT is a strided Conv1d with a
  windowed DFT basis, so every runtime that has Conv runs it (no STFT op needed).
- Encoder: features (1, T, 80) -> cross-attention keys and values for all 8 decoder layers, packed in one tensor.
- Decoder: one or more new tokens + cached state -> logits of the last token + new state. The cache holds the
  self-attention keys/values, the last 2 raw q/k/v projections (the decoder's 3-tap causal convolution), the last 9
  n-gram-memory values per site (its 4-tap convolution, dilation 3), the last 2 tokens (n-gram hashes) and the position.
check_parity.py compares every graph against the JAX model.
"""
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import load_file
from safetensors import safe_open

ENGRAM_SEED = 0x9E3779B9
ENGRAM_PRIME = 0x01000193
ENGRAM_TAPS = 4
HADA_PERM_SEEDS = (11, 13)


def native_pair(cfg):
    """(the model's own language tag, "en"). Whistle's base tags stay; fine-tunes rename one tag to their language."""
    own = [x for x in cfg["languages"] if x not in ("en", "de", "fr", "es", "it", "nl", "pl")]
    assert len(own) == 1, cfg["languages"]
    return own[0], "en"


def load(path):
    """-> (weights as float32 numpy dict, config dict, run dict) from a needle checkpoint."""
    with safe_open(path, framework="np") as h:
        meta = h.metadata()
    return load_file(path), json.loads(meta["config"]), json.loads(meta.get("run") or "{}")


def zc_norm(x, scale, eps=1e-6):
    return (1 + scale) * x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)


def rms_unit(x, eps=1e-6):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)


def rope(x, pos, theta, dim):
    """x (B, H, T, dim); pos (T,) float positions."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = pos[:, None] * freqs[None, :]
    cos, sin = torch.cos(ang), torch.sin(ang)
    h = dim // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def sinkhorn(logits, iters=20):
    for _ in range(iters):
        logits = logits - torch.logsumexp(logits, -1, keepdim=True)
        logits = logits - torch.logsumexp(logits, -2, keepdim=True)
    return torch.exp(logits)


def mel_matrix(cfg):
    def to_mel(f):
        return np.where(f < 1000.0, f * 3.0 / 200.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) * 27.0 / np.log(6.4))

    def to_hz(m):
        return np.where(m < 15.0, m * 200.0 / 3.0, 1000.0 * np.exp((m - 15.0) * np.log(6.4) / 27.0))

    points = to_hz(np.linspace(0.0, to_mel(np.array(cfg["rate"] / 2.0)), cfg["mels"] + 2))
    freqs = np.linspace(0.0, cfg["rate"] / 2, cfg["fft"] // 2 + 1)[:, None]
    lo, mid, hi = points[:-2], points[1:-1], points[2:]
    bank = np.maximum(0.0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
    return (bank * 2.0 / (hi - lo)).astype(np.float32)


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


class Frontend(torch.nn.Module):
    """jax.scipy.signal.stft(boundary='zeros', padded=False) * window.sum(), power, mel, log, per-clip mean/std."""

    def __init__(self, cfg):
        super().__init__()
        self.win, self.hop, self.fft = cfg["win"], cfg["hop"], cfg["fft"]
        window = np.hanning(self.win).astype(np.float64)
        n = np.arange(self.win)[None, :]
        k = np.arange(self.fft // 2 + 1)[:, None]
        ang = 2 * np.pi * k * n / self.fft
        basis = np.concatenate([np.cos(ang) * window, -np.sin(ang) * window], 0)  # (2*257, 400)
        self.register_buffer("basis", t(basis[:, None, :]))
        self.register_buffer("mel", t(mel_matrix(cfg)))

    def forward(self, wav):  # (1, N) float32
        n = wav.shape[1]
        x = F.pad(wav[:, None, :], (self.win // 2, self.win // 2))
        z = F.conv1d(x, self.basis, stride=self.hop)  # (1, 514, frames)
        bins = self.fft // 2 + 1
        power = z[:, :bins] ** 2 + z[:, bins:] ** 2
        x = torch.log(power.transpose(1, 2) @ self.mel + 2.0 ** -24)  # (1, frames, 80)
        valid = torch.div(torch.tensor(n), self.hop, rounding_mode="floor") if not torch.jit.is_tracing() else n // self.hop
        frames = x.shape[1]
        mask = (torch.arange(frames) < valid).to(x.dtype)[None, :, None]
        cnt = torch.clamp(torch.as_tensor(valid, dtype=x.dtype), min=1.0)
        mean = (x * mask).sum(1, keepdim=True) / cnt
        std = torch.sqrt((((x - mean) * mask) ** 2).sum(1, keepdim=True) / torch.clamp(cnt - 1, min=1.0))
        return (x - mean) / (std + 1e-5) * mask


class HadamardMLP(torch.nn.Module):
    def __init__(self, w, prefix, layer):
        super().__init__()
        g = lambda name: t(w[f"{prefix}/{name}"][layer])  # noqa: E731
        for name in ("d1", "d2", "b2", "d3", "d4", "cond_v", "cond_u", "w1a", "w1b", "w2a", "w2b", "w3a", "w3b"):
            self.register_buffer(name, g(name))
        n = self.d1.shape[0]
        self.register_buffer("p1", torch.from_numpy(np.random.RandomState(HADA_PERM_SEEDS[0]).permutation(n)))
        self.register_buffer("p2", torch.from_numpy(np.random.RandomState(HADA_PERM_SEEDS[1]).permutation(n)))

    @staticmethod
    def kron(z, a, b):
        lead = z.shape[:-1]
        z = z.reshape(*lead, a.shape[0], b.shape[0])
        z = a.transpose(0, 1) @ z @ b  # einsum("...ij,ik,jl->...kl")
        return z.reshape(*lead, a.shape[0] * b.shape[0])

    def forward(self, x):
        cond = 1 + torch.softmax(x @ self.cond_v, -1) @ self.cond_u
        z = self.kron(self.d1 * x, self.w1a, self.w1b)[..., self.p1]
        z = self.kron(F.silu(self.d2 * cond * z + self.b2), self.w2a, self.w2b)[..., self.p2]
        z = self.kron(self.d3 * z, self.w3a, self.w3b)
        return self.d4 * z


class LaneMix(torch.nn.Module):
    """mHC lane mixing for one layer (needle _LaneBody): read a block input from 4 lanes, write its output back."""

    def __init__(self, w, prefix, layer, lanes):
        super().__init__()
        g = lambda name: t(w[f"{prefix}/{name}"][layer])  # noqa: E731
        self.n = lanes
        self.register_buffer("phi", torch.cat([g("mhc_phi_pre"), g("mhc_phi_post"), g("mhc_phi_res")], -1))
        lane = np.eye(lanes, dtype=np.float32)[layer % lanes]
        self.register_buffer("b_pre", g("mhc_b_pre") + t(8 * lane - 4))
        self.register_buffer("b_post", g("mhc_b_post") + t(-4 * (1 - lane)))
        self.register_buffer("b_res", g("mhc_b_res"))
        self.a_pre, self.a_post, self.a_res = (float(w[f"{prefix}/mhc_a_{k}"][layer]) for k in ("pre", "post", "res"))

    def forward(self, x, block):  # x (B, T, n, C)
        B, T, n, C = x.shape
        p = rms_unit(x.reshape(B, T, n * C)) @ self.phi
        hpre = torch.sigmoid(self.a_pre * p[..., :n] + self.b_pre)
        hpost = 2 * torch.sigmoid(self.a_post * p[..., n:2 * n] + self.b_post)
        hres = sinkhorn(self.a_res * p[..., 2 * n:].reshape(B, T, n, n) + self.b_res)
        u = (hpre[..., None] * x).sum(2)
        y = block(u) - u
        return hres @ x + hpost[..., None] * y[:, :, None, :]


class EncoderBlock(torch.nn.Module):
    def __init__(self, w, cfg, layer):
        super().__init__()
        p = "encoder/layers/block"
        g = lambda name: t(w[f"{p}/{name}"][layer])  # noqa: E731
        self.heads, self.kv, self.qk, self.vd = cfg["dim"] // cfg["v_dim"], cfg["kv_heads"], cfg["qk_dim"], cfg["v_dim"]
        self.theta = cfg["rope"]
        for name in ("pre_hada_norm_0/scale", "ZCRMSNorm_0/scale", "post_attn_norm/scale", "conv_norm/scale",
                     "conv_out_norm/scale", "pre_hada_norm/scale", "pw1/kernel", "pw2/kernel",
                     "self_attn/q_proj/kernel", "self_attn/k_proj/kernel", "self_attn/v_proj/kernel",
                     "self_attn/gate_proj/kernel", "self_attn/out_proj/kernel", "self_attn/q_norm/scale",
                     "self_attn/k_norm/scale"):
            self.register_buffer(name.replace("/", "_"), g(name))
        self.attn_gate = float(1 / (1 + np.exp(-w[f"{p}/attn_gate"][layer])))
        self.register_buffer("dw", t(w[f"{p}/dw"][layer][:, 0, :].T[:, None, :]))  # (512, 1, 9)
        self.mlp0 = HadamardMLP(w, f"{p}/hadamard_mlp_0", layer)
        self.mlp = HadamardMLP(w, f"{p}/hadamard_mlp", layer)

    def attn(self, x, key_ok, pos):
        B, T, _ = x.shape
        q = (x @ self.self_attn_q_proj_kernel).reshape(B, T, self.heads, self.qk).transpose(1, 2)
        k = (x @ self.self_attn_k_proj_kernel).reshape(B, T, self.kv, self.qk).transpose(1, 2)
        v = (x @ self.self_attn_v_proj_kernel).reshape(B, T, self.kv, self.vd).transpose(1, 2)
        q = rope(zc_norm(q, self.self_attn_q_norm_scale), pos, self.theta, self.qk)
        k = rope(zc_norm(k, self.self_attn_k_norm_scale), pos, self.theta, self.qk)
        g = self.heads // self.kv
        q = q.reshape(B, self.kv, g, T, self.qk)
        s = q @ k[:, :, None].transpose(-1, -2) / math.sqrt(self.qk)  # (B, kv, g, T, T)
        s = s.masked_fill(~key_ok[:, None, None, None, :], torch.finfo(torch.float32).min)
        o = torch.softmax(s, -1) @ v[:, :, None]  # (B, kv, g, T, vd)
        o = o.reshape(B, self.heads, T, self.vd).transpose(1, 2).reshape(B, T, self.heads * self.vd)
        o = o * torch.sigmoid(x @ self.self_attn_gate_proj_kernel)
        return o @ self.self_attn_out_proj_kernel

    def forward(self, x, key_ok, valid, pos):
        x = x + 0.5 * self.mlp0(zc_norm(x, self.pre_hada_norm_0_scale))
        h = self.attn(zc_norm(x, self.ZCRMSNorm_0_scale), key_ok, pos)
        x = x + self.attn_gate * zc_norm(h, self.post_attn_norm_scale)
        h = zc_norm(x, self.conv_norm_scale) @ self.pw1_kernel
        a, b = h.chunk(2, -1)
        h = a * torch.sigmoid(b) * valid
        h = F.conv1d(h.transpose(1, 2), self.dw, padding=self.dw.shape[-1] // 2, groups=h.shape[-1]).transpose(1, 2)
        h = F.silu(zc_norm(h, self.conv_out_norm_scale))
        x = x + h @ self.pw2_kernel
        return x + 0.5 * self.mlp(zc_norm(x, self.pre_hada_norm_scale))


class Encoder(torch.nn.Module):
    """features (1, T, 80) -> packed cross keys/values (1, F, L * H * (qk + v)) for F valid encoder frames."""

    def __init__(self, w, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("stem_w", t(np.transpose(w["stem/w"], (3, 2, 0, 1))))  # (128, 1, 3, 3)
        for i in (1, 2):
            self.register_buffer(f"stem_dw_{i}", t(np.transpose(w[f"stem/dw_{i}"], (3, 2, 0, 1))))
            self.register_buffer(f"stem_pw_{i}", t(w[f"stem/pw_{i}/kernel"]))
        self.register_buffer("stem_out", t(w["stem/out/kernel"]))
        self.blocks = torch.nn.ModuleList(EncoderBlock(w, cfg, i) for i in range(cfg["enc_layers"]))
        self.mix = torch.nn.ModuleList(LaneMix(w, "encoder", i, cfg["enc_lanes"]) for i in range(cfg["enc_layers"]))
        self.register_buffer("final_norm", t(w["encoder/final_norm/scale"]))
        self.pe_gate = float(1 / (1 + np.exp(-float(w["pe_gate"]))))
        # cross-attention projections of all decoder layers, applied once per clip
        p = "stack/layers/block/cross_attn"
        L = cfg["dec_layers"]
        cat = lambda a: t(np.concatenate(list(a), -1))  # noqa: E731  (L, 512, e) -> (512, L*e): one MatMul
        self.register_buffer("ck", cat(w[f"{p}/k_proj/kernel"]))  # (512, L*384)
        self.register_buffer("cv", cat(w[f"{p}/v_proj/kernel"]))  # (512, L*512)
        self.register_buffer("ck_norm", t(w[f"{p}/k_norm/scale"][:, None, :]))  # (L, 1, 48)
        self.heads, self.qk, self.vd, self.L = cfg["dim"] // cfg["v_dim"], cfg["qk_dim"], cfg["v_dim"], L

    def stem(self, x, n):
        """x (1, T, 80) -> (1, T', 512); n valid frames (int tensor) -> valid frames after 3 stride-2 stages."""
        x = x[:, None]  # (1, 1, T, 80)

        def down(x, wt, n, groups):
            x = F.conv2d(x, wt, stride=2, padding=1, groups=groups)
            n = torch.div(n + 1, 2, rounding_mode="floor")
            return x * (torch.arange(x.shape[2]) < n).to(x.dtype)[None, None, :, None], n

        x, n = down(x, self.stem_w, n, 1)
        for i in (1, 2):
            x, n = down(F.silu(x), getattr(self, f"stem_dw_{i}"), n, x.shape[1])
            x = (x.permute(0, 2, 3, 1) @ getattr(self, f"stem_pw_{i}")).permute(0, 3, 1, 2)
        x = F.silu(x).permute(0, 2, 1, 3)  # (1, T', 128, F')
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x @ self.stem_out, n

    def encode(self, feats, n):
        x, n = self.stem(feats, n)
        x = F.pad(x, (0, 0, 0, x.shape[1] % 2))
        T = x.shape[1]
        seen = torch.arange(T)[None, :] < n
        pos = torch.arange(T, dtype=torch.float32)
        x = x[:, :, None, :].expand(-1, -1, self.cfg["enc_lanes"], -1)
        for blk, mix in zip(self.blocks, self.mix):
            x = mix(x, lambda u, blk=blk: blk(u, seen, seen[..., None].to(u.dtype), pos))
        return zc_norm(x.mean(2), self.final_norm), n

    def memory(self, enc):
        T = enc.shape[1]
        d = enc.shape[-1]
        ang = torch.arange(T, dtype=torch.float32)[:, None] * torch.exp(
            -math.log(10000.0) / (d // 2 - 1) * torch.arange(d // 2, dtype=torch.float32))
        return enc + self.pe_gate * torch.cat([torch.sin(ang), torch.cos(ang)], -1)

    def forward(self, feats, n):
        """feats (1, T, 80); n (1,) int64 valid feature frames -> cross (1, F, L*H*(qk+v))."""
        enc, n = self.encode(feats, n)
        mem = self.memory(enc)
        mem = mem[:, : n[0]] if not torch.jit.is_tracing() else mem.narrow(1, 0, n[0])
        B, Fr, _ = mem.shape
        k = zc_norm((mem @ self.ck).reshape(B, Fr, self.L, self.heads, self.qk), self.ck_norm)
        v = mem @ self.cv
        # packed per frame: [layer][k heads*qk | v heads*vd]
        k = k.reshape(B, Fr, self.L, self.heads * self.qk)
        v = v.reshape(B, Fr, self.L, self.heads * self.vd)
        return torch.cat([k, v], -1).reshape(B, Fr, -1)


class DecoderBlock(torch.nn.Module):
    def __init__(self, w, cfg, layer):
        super().__init__()
        p = "stack/layers/block"
        g = lambda name: t(w[f"{p}/{name}"][layer])  # noqa: E731
        self.heads, self.kv, self.qk, self.vd = cfg["dim"] // cfg["v_dim"], cfg["kv_heads"], cfg["qk_dim"], cfg["v_dim"]
        self.theta = cfg["rope"]
        sa = "self_attn"
        self.register_buffer("wqkv", torch.cat([g(f"{sa}/q_proj/kernel"), g(f"{sa}/k_proj/kernel"),
                                                g(f"{sa}/v_proj/kernel")], -1))  # (512, 608)
        self.register_buffer("taps", torch.cat([g(f"{sa}/q_taps"), g(f"{sa}/k_taps"), g(f"{sa}/v_taps")], -1))  # (3, 608)
        for name in ("ZCRMSNorm_0/scale", "post_attn_norm/scale", "cross_norm/scale", "post_cross_norm/scale",
                     "pre_hada_norm/scale", f"{sa}/q_norm/scale", f"{sa}/k_norm/scale", f"{sa}/gate_proj/kernel",
                     f"{sa}/out_proj/kernel", "cross_attn/q_proj/kernel", "cross_attn/q_norm/scale",
                     "cross_attn/gate_proj/kernel", "cross_attn/out_proj/kernel"):
            self.register_buffer(name.replace("/", "_"), g(name))
        sig = lambda v: float(1 / (1 + np.exp(-v)))  # noqa: E731
        self.attn_gate, self.cross_gate = sig(w[f"{p}/attn_gate"][layer]), sig(w[f"{p}/cross_gate"][layer])
        self.mlp = HadamardMLP(w, f"{p}/hadamard_mlp", layer)
        self.ntaps = self.taps.shape[0]

    def self_attn(self, x, st, pos):
        """x (B, S, 512); st dict with this layer's cache: tap (B, 2, 608), k (B, kv, P, qk), v (B, kv, P, vd)."""
        B, S, _ = x.shape
        raw = torch.cat([st["tap"], x @ self.wqkv], 1)  # (B, 2 + S, 608)
        nt = self.ntaps
        mixed = sum(self.taps[j] * raw[:, nt - 1 - j: nt - 1 - j + S] for j in range(nt))
        st["tap"] = raw[:, -(nt - 1):]
        hq, hk = self.heads * self.qk, self.kv * self.qk
        q = mixed[..., :hq].reshape(B, S, self.heads, self.qk).transpose(1, 2)
        k = mixed[..., hq:hq + hk].reshape(B, S, self.kv, self.qk).transpose(1, 2)
        v = mixed[..., hq + hk:].reshape(B, S, self.kv, self.vd).transpose(1, 2)
        q = rope(zc_norm(q, self.self_attn_q_norm_scale), pos, self.theta, self.qk)
        k = rope(zc_norm(k, self.self_attn_k_norm_scale), pos, self.theta, self.qk)
        k = torch.cat([st["k"], k], 2)
        v = torch.cat([st["v"], v], 2)
        st["k"], st["v"] = k, v
        P = k.shape[2]
        g = self.heads // self.kv
        q = q.reshape(B, self.kv, g, S, self.qk)
        s = q @ k[:, :, None].transpose(-1, -2) / math.sqrt(self.qk)  # (B, kv, g, S, P)
        ok = torch.arange(P)[None, :] <= (torch.arange(S)[:, None] + (P - S))
        s = s.masked_fill(~ok, torch.finfo(torch.float32).min)
        o = (torch.softmax(s, -1) @ v[:, :, None]).reshape(B, self.heads, S, self.vd)
        o = o.transpose(1, 2).reshape(B, S, self.heads * self.vd)
        o = o * torch.sigmoid(x @ self.self_attn_gate_proj_kernel)
        return o @ self.self_attn_out_proj_kernel

    def cross_attn(self, x, ck, cv):
        """ck (1, H, F, qk) normed keys, cv (1, H, F, vd)."""
        B, S, _ = x.shape
        q = (x @ self.cross_attn_q_proj_kernel).reshape(B, S, self.heads, self.qk).transpose(1, 2)
        q = zc_norm(q, self.cross_attn_q_norm_scale)
        s = q @ ck.transpose(-1, -2) / math.sqrt(self.qk)
        o = (torch.softmax(s, -1) @ cv).transpose(1, 2).reshape(B, S, self.heads * self.vd)
        o = o * torch.sigmoid(x @ self.cross_attn_gate_proj_kernel)
        return o @ self.cross_attn_out_proj_kernel

    def forward(self, x, st, pos, ck, cv, engram=None):
        if engram is not None:
            ek, ev = engram
            alpha = torch.sigmoid((rms_unit(x) * rms_unit(ek)).sum(-1, keepdim=True) / math.sqrt(x.shape[-1]))
            x = x + alpha * ev
        h = self.self_attn(zc_norm(x, self.ZCRMSNorm_0_scale), st, pos)
        x = x + self.attn_gate * zc_norm(h, self.post_attn_norm_scale)
        h = self.cross_attn(zc_norm(x, self.cross_norm_scale), ck, cv)
        x = x + self.cross_gate * zc_norm(h, self.post_cross_norm_scale)
        return x + self.mlp(zc_norm(x, self.pre_hada_norm_scale))


class Engram(torch.nn.Module):
    """Hashed n-gram memory of one site: token n-grams -> table rows -> key and value (with a dilated 4-tap conv)."""

    def __init__(self, w, cfg, site):
        super().__init__()
        p = f"engrams_{site}"
        tables = w[f"{p}/embedding"]  # (heads*orders, slots, sub)
        self.nt, self.slots, self.sub = tables.shape
        self.register_buffer("tables", t(tables.reshape(-1, self.sub)))
        self.register_buffer("key_proj", t(w[f"{p}/key_proj/kernel"]))
        self.register_buffer("value_proj", t(w[f"{p}/value_proj/kernel"]))
        self.register_buffer("taps", t(w[f"{p}/taps"]))  # (4, 512)
        self.dil = max(cfg["orders"])

    def forward(self, idx, ok, vcache):
        """idx (B, S, nt) int64 row indices; ok (B or 1, S, nt) float; vcache (B, 9, 512) last raw values."""
        rows = idx + torch.arange(self.nt)[None, None, :] * self.slots
        e = F.embedding(rows, self.tables) * ok[..., None]  # (B, S, nt, sub)
        e = e.reshape(e.shape[0], e.shape[1], self.nt * self.sub)
        k = e @ self.key_proj
        S = e.shape[1]
        raw = torch.cat([vcache, e @ self.value_proj], 1)  # (B, 9 + S, 512)
        W = (ENGRAM_TAPS - 1) * self.dil
        v = sum(self.taps[j] * raw[:, W - j * self.dil: W - j * self.dil + S] for j in range(ENGRAM_TAPS))
        return k, v, raw[:, -W:]


POW2 = torch.tensor([1 << i for i in range(32)], dtype=torch.int64)


def idiv(a, b):
    """Integer division of non-negative int64 values: ONNX Div (it truncates, which equals floor here). Not
    rounding_mode="floor": its export adds sign fix-ups that came out wrong once shapes were static (sherpa graphs)."""
    return torch.div(a, b, rounding_mode="trunc")


def imod(a, b):
    return a - idiv(a, b) * b


def xor32(a, b):
    """Bitwise xor of non-negative int64 values below 2^32 with Div/Mul/Sub/ReduceSum only, which every ONNX runtime
    has for int64 (bitwise ops are missing from older and mobile builds)."""
    ba = imod(idiv(a[..., None], POW2), 2)
    bb = imod(idiv(b[..., None], POW2), 2)
    return (imod(ba + bb, 2) * POW2).sum(-1)


def engram_indices(hist, tokens, orders, heads, slots):
    """hist (B, 2) previous tokens (0 before the start), tokens (B, S) -> (B, S, len(orders)*heads) int64 rows.
    needle's uint32 hash (xor, multiply mod 2^32, xor-shift 15, mod slots) done in int64 arithmetic."""
    full = torch.cat([hist, tokens], 1).to(torch.int64)
    S = tokens.shape[1]
    H = hist.shape[1]
    out = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (ENGRAM_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF
            # tokens * 0, not zeros_like: with static shapes the exporter folds a constant seed through the divisions
            # in float and gets the hash wrong; a runtime value keeps the int64 arithmetic in the runtime
            acc = full[:, H:] * 0 + seed
            for j in range(order):
                u = full[:, H - j: H - j + S]
                acc = imod(xor32(acc, u) * ENGRAM_PRIME, 1 << 32)
            acc = xor32(acc, idiv(acc, 32768))
            out.append(imod(acc, slots))
    return torch.stack(out, -1)


class Decoder(torch.nn.Module):
    """Incremental decoder. External token ids swap needle's bos (2) and eos (1) when swap_bos_eos is set, so the
    graph follows the sos=1 / eos=2 convention of sherpa-onnx's Moonshine runner; language tags never win the argmax
    after the prompt (mask_tags)."""

    def __init__(self, w, cfg, swap_bos_eos=False, mask_tags=True, first_tags=None):
        """first_tags: language codes (e.g. ("hi", "en")). When set, the first call (position 0) ignores its token and
        reads bos instead, and returns logits over those language tags only, so a runner that only knows "feed sos,
        then feed back the argmax" (sherpa-onnx's Moonshine loop) detects the language before the text."""
        super().__init__()
        self.cfg = cfg
        self.L, self.lanes, self.dim = cfg["dec_layers"], cfg["lanes"], cfg["dim"]
        self.heads, self.kv, self.qk, self.vd = cfg["dim"] // cfg["v_dim"], cfg["kv_heads"], cfg["qk_dim"], cfg["v_dim"]
        self.register_buffer("emb", t(w["embedding/embedding"]))
        self.blocks = torch.nn.ModuleList(DecoderBlock(w, cfg, i) for i in range(self.L))
        self.mix = torch.nn.ModuleList(LaneMix(w, "stack", i, self.lanes) for i in range(self.L))
        self.engrams = torch.nn.ModuleList(Engram(w, cfg, s) for s in range(len(cfg["sites"])))
        self.sites = {layer: s for s, layer in enumerate(cfg["sites"])}
        self.register_buffer("final_norm", t(w["stack/final_norm/scale"]))
        V = self.emb.shape[0]
        perm = np.arange(V)
        if swap_bos_eos:
            perm[[cfg["bos"], cfg["eos"]]] = perm[[cfg["eos"], cfg["bos"]]]
        self.register_buffer("perm", torch.from_numpy(perm))  # external id -> internal id (an involution)
        bias = np.zeros(V, np.float32)
        if mask_tags:
            bias[cfg["text"]:] = -1e9
        self.register_buffer("logit_bias", t(bias[perm]))
        self.first = first_tags is not None
        if self.first:
            fb = np.full(V, -1e9, np.float32)
            for lang in first_tags:
                fb[cfg["text"] + cfg["languages"].index(lang)] = 0.0
            self.register_buffer("first_bias", t(fb[perm]))
            self.bos_id = cfg["bos"]
        self.orders, self.hashes, self.slots = tuple(cfg["orders"]), cfg["hashes"], cfg["slots"]
        self.ntaps = cfg["taps"]
        self.ew = (ENGRAM_TAPS - 1) * max(self.orders)

    def init_state(self, B=1):
        z = lambda *s: torch.zeros(*s)  # noqa: E731
        st = {"pos": torch.zeros(1, dtype=torch.int64), "hist": torch.zeros(B, max(self.orders) - 1, dtype=torch.int64)}
        for i in range(self.L):
            st[f"tap{i}"] = z(B, self.ntaps - 1, self.wqkv_width)
            st[f"k{i}"] = z(B, self.kv, 0, self.qk)
            st[f"v{i}"] = z(B, self.kv, 0, self.vd)
        for s in range(len(self.engrams)):
            st[f"ev{s}"] = z(B, self.ew, self.dim)
        return st

    @property
    def wqkv_width(self):
        return self.blocks[0].wqkv.shape[1]

    def state_names(self):
        names = ["pos", "hist"]
        for i in range(self.L):
            names += [f"tap{i}", f"k{i}", f"v{i}"]
        return names + [f"ev{s}" for s in range(len(self.engrams))]

    def forward(self, tokens, cross, *state):
        """tokens (B, S) external ids; cross (1, F, L*H*(qk+v)) from the encoder; state in state_names() order.
        -> logits of the last token (B, 1, V) in external ids, then the new state in the same order."""
        st = dict(zip(self.state_names(), state))
        tok = self.perm[tokens]
        B, S = tok.shape
        pos0 = st["pos"]
        if self.first:  # integer arithmetic, not where/full_like: the exporter typed that branch as float, and the
            is0 = (pos0[:, None] == 0).to(torch.int64)  # n-gram hash then lost precision
            tok = tok * (1 - is0) + self.bos_id * is0
        pos = (pos0 + torch.arange(S)).to(torch.float32)
        # -1, not cross.shape[1]: the exporter can freeze a shape read from a reshaped input to the traced length
        c = cross.reshape(1, -1, self.L, self.heads * (self.qk + self.vd))
        ck = c[..., : self.heads * self.qk].reshape(1, -1, self.L, self.heads, self.qk).permute(2, 0, 3, 1, 4)
        cv = c[..., self.heads * self.qk:].reshape(1, -1, self.L, self.heads, self.vd).permute(2, 0, 3, 1, 4)
        x = self.emb[tok] * math.sqrt(self.dim)
        # n-gram memory: rows valid once the n-gram fits (position >= order - 1)
        idx = engram_indices(st["hist"], tok, self.orders, self.hashes, self.slots)
        ok = torch.stack([(pos >= o - 1).to(torch.float32) for o in self.orders for _ in range(self.hashes)], -1)[None]
        kv = []
        for s, e in enumerate(self.engrams):
            k, v, st[f"ev{s}"] = e(idx, ok, st[f"ev{s}"])
            kv.append((k, v))
        st["hist"] = torch.cat([st["hist"], tok], 1)[:, -(max(self.orders) - 1):]
        x = x[:, :, None, :].expand(-1, -1, self.lanes, -1)
        for i, (blk, mix) in enumerate(zip(self.blocks, self.mix)):
            layer_st = {"tap": st[f"tap{i}"], "k": st[f"k{i}"], "v": st[f"v{i}"]}
            eng = kv[self.sites[i]] if i in self.sites else None
            x = mix(x, lambda u, blk=blk, layer_st=layer_st, eng=eng, i=i: blk(u, layer_st, pos, ck[i], cv[i], eng))
            st[f"tap{i}"], st[f"k{i}"], st[f"v{i}"] = layer_st["tap"], layer_st["k"], layer_st["v"]
        h = zc_norm(x[:, -1].mean(1), self.final_norm)  # (B, 512)
        logits = F.linear(h, self.emb)  # Gemm with transB: shares the embedding table, no transposed copy
        bias = torch.where(pos0[:, None] == 0, self.first_bias, self.logit_bias) if self.first else self.logit_bias
        logits = (logits[:, self.perm] + bias)[:, None, :]
        st["pos"] = pos0 + S
        return (logits, *[st[n] for n in self.state_names()])


def build(path, swap_bos_eos=False, mask_tags=True):
    w, cfg, run = load(path)
    return Frontend(cfg).eval(), Encoder(w, cfg).eval(), Decoder(w, cfg, swap_bos_eos, mask_tags).eval(), cfg, run
