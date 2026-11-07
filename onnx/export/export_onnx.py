"""Export Shruti's float ONNX graphs (frontend, encoder, decoder) and check them against PyTorch in ONNX Runtime.
Usage: python export_onnx.py <full_model.safetensors> <out_dir> <check_clip.wav> [--sherpa]
--sherpa: decoder ids follow sherpa-onnx's Moonshine convention (sos=1, eos=2: needle's bos/eos swapped)."""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch

from quantize_onnx import share_tied_embedding
from shruti_torch import build

ckpt, out_dir, clip = sys.argv[1], sys.argv[2], sys.argv[3]
sherpa = "--sherpa" in sys.argv
os.makedirs(out_dir, exist_ok=True)
fe, enc, dec, cfg, run = build(ckpt, swap_bos_eos=sherpa, mask_tags=sherpa)
OPSET = 18


class EncoderFromFeats(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, feats, n):  # n (1,) int64 valid feature frames
        return self.enc(feats, n)


wav, _ = sf.read(clip, dtype="float32")
wav_t = torch.from_numpy(wav[None])
with torch.no_grad():
    feats = fe(wav_t)
    n = torch.tensor([len(wav) // cfg["hop"]])
    cross = enc(feats, n)
    st0 = dec.init_state(2)
    bos = 1 if sherpa else cfg["bos"]
    tok = torch.tensor([[bos, cfg["text"] + 7, 100], [bos, cfg["text"] + 7, 200]])
    out = dec(tok, cross, *st0.values())  # 3 tokens in: dummy state with a non-empty cache for tracing
    st3 = list(out[1:])

names = dec.state_names()

# vocabulary for detokenising: needle's base sentencepiece pieces, then our 714 added Devanagari pieces
import json  # noqa: E402
import sentencepiece as spm  # noqa: E402
base = spm.SentencePieceProcessor(model_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "../whistle/needle_src/needle/model/tokenizer.model"))
vocab = [base.id_to_piece(i) for i in range(base.get_piece_size())] + list(run["pieces"])
assert len(vocab) == cfg["text"], (len(vocab), cfg["text"])
vocab += [f"<|{lang}|>" for lang in cfg["languages"]]
json.dump({"config": cfg, "vocab": vocab, "sherpa_ids": sherpa}, open(f"{out_dir}/shruti.json", "w", encoding="utf-8"),
          ensure_ascii=False)
kw = dict(opset_version=OPSET, dynamo=False, do_constant_folding=True)
torch.onnx.export(fe, (wav_t,), f"{out_dir}/frontend.onnx", input_names=["audio"], output_names=["features"],
                  dynamic_axes={"audio": {1: "samples"}, "features": {1: "frames"}}, **kw)
torch.onnx.export(EncoderFromFeats(enc), (feats, n), f"{out_dir}/encoder.onnx", input_names=["features", "frames"],
                  output_names=["cross"], dynamic_axes={"features": {1: "frames"}, "cross": {1: "enc_frames"}}, **kw)
dyn = {"tokens": {0: "batch", 1: "new"}, "cross": {1: "enc_frames"}, "logits": {0: "batch"}}
in_names = ["tokens", "cross"] + names
out_names = ["logits"] + [f"new_{s}" for s in names]
for s in names:
    if s == "pos":
        continue
    ax = {0: "batch"}
    if s[0] in "kv" and s[1:].isdigit():
        ax[2] = "past"
    dyn[s] = ax
    dyn[f"new_{s}"] = {0: "batch", 2: "past_new"} if 2 in ax else {0: "batch"}
torch.onnx.export(dec, (tok[:, 2:], cross, *st3), f"{out_dir}/decoder.onnx", input_names=in_names,
                  output_names=out_names, dynamic_axes=dyn, **kw)

share_tied_embedding(f"{out_dir}/decoder.onnx")
for f in ("frontend", "encoder", "decoder"):
    m = onnx.load(f"{out_dir}/{f}.onnx")
    ops = sorted({nd.op_type for nd in m.graph.node})
    print(f, f"{os.path.getsize(f'{out_dir}/{f}.onnx') / 1e6:.1f} MB", "ops:", " ".join(ops))

# ---- ONNX Runtime vs PyTorch, on shapes different from the trace (odd lengths, batch 1 and 5, empty cache) ----
so = ort.SessionOptions()
sess = {f: ort.InferenceSession(f"{out_dir}/{f}.onnx", so, providers=["CPUExecutionProvider"])
        for f in ("frontend", "encoder", "decoder")}


def rel(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


for cut in (len(wav), len(wav) - 4321, 16000 * 3 + 77):
    w = wav[:cut][None]
    with torch.no_grad():
        f_t = fe(torch.from_numpy(w)).numpy()
        n_t = np.array([cut // cfg["hop"]], np.int64)
        c_t = enc(torch.from_numpy(f_t), torch.from_numpy(n_t)).numpy()
    f_o = sess["frontend"].run(None, {"audio": w})[0]
    c_o = sess["encoder"].run(None, {"features": f_o, "frames": n_t})[0]
    print(f"samples {cut}: features rel {rel(f_o, f_t):.1e} {f_o.shape} | cross rel {rel(c_o, c_t):.1e} {c_o.shape}")
    for B in (1, 5):
        st = {k: v.numpy() for k, v in dec.init_state(B).items()}
        toks = np.array([[bos, cfg["text"] + 7]] * B, np.int64)
        lt, lo = [], []
        st_t = [torch.from_numpy(st[k]) for k in names]
        st_o = [st[k] for k in names]
        for step in range(6):
            with torch.no_grad():
                r_t = dec(torch.from_numpy(toks), torch.from_numpy(c_t), *st_t)
            r_o = sess["decoder"].run(None, {"tokens": toks, "cross": c_o, **dict(zip(names, st_o))})
            lt.append(r_t[0].numpy())
            lo.append(r_o[0])
            st_t, st_o = list(r_t[1:]), list(r_o[1:])
            toks = r_t[0].numpy()[:, 0].argmax(-1)[:, None].astype(np.int64)
        lt, lo = np.stack(lt)[..., :cfg["text"]], np.stack(lo)[..., :cfg["text"]]  # text logits (tags are -1e9)
        print(f"  batch {B}: decoder logits rel {rel(lo, lt):.1e} argmax agree {(lo.argmax(-1) == lt.argmax(-1)).mean():.2f}")
