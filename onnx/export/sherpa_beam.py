"""Beam search inside the decoder graph, for runners that only feed back the argmax (sherpa-onnx's Moonshine loop).

Each call runs one decoder step for all K beams on the beams' own next tokens, which the graph keeps in its state
(sherpa's token input is ignored). The returned "logits" are one-hot: a placeholder id while the search runs (the
'de' language tag, absent from tokens.txt, so it never prints), then the winning hypothesis one token per call, then
eos. The search follows shruti_onnx.Shruti._transcribe: language detected over hi/en after bos, length-normalised
scores (sum of log-probabilities / (tokens + 1)), the 2K best candidates per step, stop once K hypotheses have ended
and no live beam can beat the best of them, at most 12 tokens a second + 8 (and 318). The Python runner's token-loop
cut is not done here.

Calls: 1) bos -> pick the language tag; 2) tag -> the K best first tokens; 3...) one search step each; then emit.
Token columns line up with calls: column 0 is padding and a hypothesis's tokens start at column 1, so every
sequence tensor has as many columns as calls made.
"""
import torch

from shruti_torch import native_pair

NEG = -1e9


class BeamDecoder(torch.nn.Module):
    def __init__(self, dec, cfg, width, fold, K=5, langs=None, placeholder="de"):
        """dec: shruti_torch.Decoder with needle ids (no swap), no logit bias and no first_tags. langs: default, the
        checkpoint's own (language, "en") pair."""
        super().__init__()
        self.dec, self.width, self.fold, self.K = dec, width, fold, K
        langs = langs or native_pair(cfg)
        V = dec.emb.shape[0]
        self.V, self.bos, self.eos = V, cfg["bos"], cfg["eos"]
        self.register_buffer("tags", torch.tensor([cfg["text"] + cfg["languages"].index(x) for x in langs]))
        mask = torch.zeros(V)
        mask[cfg["text"]:] = NEG
        self.register_buffer("text_mask", mask)
        out_map = torch.arange(V)
        out_map[[cfg["bos"], cfg["eos"]]] = out_map[[cfg["eos"], cfg["bos"]]]
        self.register_buffer("out_map", out_map)  # needle id -> sherpa id (bos/eos swapped)
        self.placeholder = cfg["text"] + cfg["languages"].index(placeholder)
        self.max_tokens = cfg["max_tokens"] - 2

    def state_names(self):
        return self.dec.state_names() + ["feed", "scores", "seqs", "nfin", "best_score", "best_seq", "best_len",
                                         "phase", "emit", "out_seq", "out_len"]

    def init_state(self):
        K = self.K
        st = self.dec.init_state(K)
        z = lambda *s: torch.zeros(*s, dtype=torch.int64)  # noqa: E731
        st.update({"feed": torch.full((K, 1), self.bos, dtype=torch.int64), "scores": torch.zeros(K),
                   "seqs": z(K, 0), "nfin": z(1), "best_score": torch.full((1,), -1e30), "best_seq": z(1, 0),
                   "best_len": z(1), "phase": z(1), "emit": z(1), "out_seq": z(1, 0), "out_len": z(1)})
        return st

    def forward(self, tokens, cross, seq_len, *state):
        K, V = self.K, self.V
        st = dict(zip(self.state_names(), state))
        dnames = self.dec.state_names()
        dstate = [st[n] for n in dnames]
        # sherpa's token and seq_len are not used; keep them as graph inputs
        dstate[0] = dstate[0] + 0 * seq_len.to(torch.int64) + 0 * tokens.to(torch.int64).reshape(-1)[:1]
        pos = dstate[0]  # (1,) calls made before this one
        out = self.dec(st["feed"], cross.reshape(1, -1, self.width), *dstate)
        lg = out[0][:, 0, :]  # (K, V), needle ids
        dnew = dict(zip(dnames, out[1:]))
        phase = st["phase"]
        is0, is1, is2, is3 = phase == 0, phase == 1, phase == 2, phase == 3
        ar = torch.arange(K)
        z1 = torch.zeros(1, dtype=torch.int64)
        seqs, best_seq, out_seq = st["seqs"], st["best_seq"], st["out_seq"]
        pad_k = torch.zeros(K, 1, dtype=torch.int64)
        pad_1 = torch.zeros(1, 1, dtype=torch.int64)

        # phase 0 (after bos): the language tag
        tag = self.tags[torch.argmax(lg[0][self.tags])]
        # phase 1 (after the tag): the K best first tokens; an eos among them ends an empty hypothesis
        logp = torch.log_softmax(lg + self.text_mask, -1)
        v1, i1 = torch.topk(logp[0], K)
        e1 = i1 == self.eos
        v1_best = torch.where(e1, v1, torch.full_like(v1, -1e30)).max().reshape(1)
        # phase 2: one search step over the 2K best (beam, token) candidates
        v2, f2 = torch.topk((st["scores"][:, None] + logp).reshape(-1), 2 * K)
        b2 = torch.div(f2, V, rounding_mode="trunc")
        k2 = f2 - b2 * V
        is_e = k2 == self.eos
        valid = v2 > NEG / 2
        live = (~is_e) & valid
        cum = torch.cumsum(live.to(torch.int64), 0)
        take = live & (cum <= K)
        fin = is_e & valid & (cum < K)  # an eos before the K-th live candidate ends that hypothesis
        t_before = (pos - 1).to(torch.float32)  # tokens in every live hypothesis before this step
        norm_fin = torch.where(fin, v2 / (t_before + 1), torch.full_like(v2, -1e30))
        f_best, f_idx = norm_fin.max(0)
        f_best, f_b = f_best.reshape(1), b2[f_idx].reshape(1)
        rank = torch.where(take, cum - 1, torch.full_like(cum, -1))
        onehot = (rank[None, :] == ar[:, None]).to(torch.float32)  # (K slots, 2K candidates)
        filled = onehot.sum(1) > 0
        new_b = (onehot @ b2.to(torch.float32)).to(torch.int64)
        new_k = (onehot @ k2.to(torch.float32)).to(torch.int64)
        new_v = torch.where(filled, onehot @ v2, torch.full((K,), NEG))

        # beams: reorder by origin in phase 2, append this call's column
        src = torch.where(is2, new_b, ar)
        col = torch.where(is1, i1, torch.where(is2, new_k, torch.zeros_like(i1)))
        seqs_new = torch.cat([seqs.index_select(0, src), col[:, None]], 1)
        dnew = {n: (v if n == "pos" else v.index_select(0, src)) for n, v in dnew.items()}
        scores = torch.where(is1, torch.where(e1, torch.full_like(v1, NEG), v1), torch.where(is2, new_v, st["scores"]))
        feed = torch.where(is0, tag.reshape(1, 1).expand(K, 1),
                           torch.where(is1, i1[:, None], torch.where(is2, new_k[:, None], st["feed"])))

        # finished hypotheses: count and keep the best (score normalised by its length + 1)
        nfin = st["nfin"] + torch.where(is1, e1.sum().reshape(1), torch.where(is2, fin.sum().reshape(1), z1))
        cand_best = torch.where(is1, v1_best, torch.where(is2, f_best, torch.full((1,), -1e30)))
        improve = cand_best > st["best_score"]
        best_score = torch.where(improve, cand_best, st["best_score"])
        cand_row = torch.where(is2, seqs.index_select(0, f_b), torch.zeros_like(seqs[:1]))  # phase 1: empty
        best_seq_new = torch.where(improve, torch.cat([cand_row, pad_1], 1), torch.cat([best_seq, pad_1], 1))
        cand_len = torch.where(is2, pos - 1, z1)
        best_len = torch.where(improve, cand_len, st["best_len"])

        # stop: K hypotheses ended and no live beam can beat the best, no live beam left, or the length cap
        frames = torch.div(cross.shape[1] + 0 * pos, self.fold, rounding_mode="trunc").to(torch.float32)
        cap = torch.clamp(torch.floor(frames * 0.08 * 12.0 + 8.0), max=float(self.max_tokens)).to(torch.int64)
        live_norm = torch.where(filled, new_v / (t_before + 2), torch.full_like(new_v, -1e30)).max().reshape(1)
        any_live = filled.any().reshape(1)
        done = is2 & (((nfin >= K) & ((~any_live) | (live_norm < best_score))) | (~any_live) | (pos >= cap))
        # the answer: the best ended hypothesis, else the best live one
        live_pick = torch.argmax(torch.where(filled, new_v, torch.full_like(new_v, -1e30))).reshape(1)
        pick_row = torch.where(nfin > 0, best_seq_new, seqs_new.index_select(0, live_pick))
        pick_len = torch.where(nfin > 0, best_len, pos)
        out_seq_new = torch.where(done, pick_row, torch.cat([out_seq, pad_1], 1))
        out_len = torch.where(done, pick_len, st["out_len"])

        # what this call says: placeholder while searching, then the answer one token per call, then eos
        emitting = done | is3
        emit_i = torch.where(done, z1, st["emit"])
        col_i = 1 + torch.clamp(torch.minimum(emit_i, out_len - 1), min=0)  # no shape reads: they can freeze on export
        tok = torch.cat([out_seq_new, pad_1, pad_1], 1)[0].index_select(0, col_i)  # padded: safe before any answer
        said = torch.where(emit_i < out_len, self.out_map[tok], self.out_map[torch.full_like(z1, self.eos)])
        said = torch.where(emitting, said, torch.full_like(said, self.placeholder))
        # digital silence: the encoder returns all zeros (sherpa_export.SherpaEncoder); answer eos on the first call
        silent = cross.abs().max().reshape(1) == 0
        said = torch.where(silent, self.out_map[torch.full_like(z1, self.eos)], said)
        logits = torch.nn.functional.one_hot(said, V).to(torch.float32)[None]  # (1, 1, V)
        phase_new = torch.where(is0, z1 + 1, torch.where(is1, z1 + 2, torch.where(done, z1 + 3, phase)))
        emit = torch.where(emitting, emit_i + 1, st["emit"])

        extra = {"feed": feed, "scores": scores, "seqs": seqs_new, "nfin": nfin, "best_score": best_score,
                 "best_seq": best_seq_new, "best_len": best_len, "phase": phase_new, "emit": emit,
                 "out_seq": out_seq_new, "out_len": out_len}
        return (logits, *[dnew[n] for n in dnames], *[extra[n] for n in self.state_names()[len(dnames):]])
