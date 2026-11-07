"""Shruti graphs in the I/O layout of sherpa-onnx's Moonshine (v1) runner, so stock sherpa-onnx (Python, C++, Android,
iOS, WebAssembly) loads them with OfflineRecognizer.from_moonshine(...). No sherpa-onnx change is needed.

sherpa-onnx's loop (sherpa-onnx/csrc/offline-moonshine-*.cc, read 27 Oct 2026):
  features = preprocess(audio (1, N));  enc = encode(features, int32 [T])
  logits, *states = uncached_decode(int32 [[sos=1]], enc, int32 [1])
  repeat up to int(enc.shape[1] * 384 / 16000 * 6) times: argmax; stop at eos=2; else
      logits, *states = cached_decode(int32 [[token]], enc, int32 [seq_len], *states)
  text = concatenation of tokens.txt symbols (ids missing from tokens.txt are skipped; a leading ▁ is a space).
How Shruti fits it:
- ids are needle's with bos/eos swapped (sos=1, eos=2); tokens.txt leaves out control pieces and language tags.
- encode returns the decoder's cross-attention keys/values, reshaped (1, 16 F, 448) so sherpa's length cap
  (0.144 tokens per row) allows 28.8 tokens a second instead of 1.8; the decoder reshapes it back.
- uncached_decode is a 1 KB graph: constant logits (its argmax is a tag missing from tokens.txt) and an empty state.
- the first cached_decode call reads bos instead of that token and returns logits over the language and en tags only, so the
  model detects the language (it is skipped in the text); later calls return text logits with the tags masked.
Usage (onnx venv): python sherpa_export.py <full_model.safetensors> <out_dir> <check_clip.wav> [--beam 5]
--beam K: 5-beam search inside the cached-decoder graph (sherpa_beam.py), the engine's decoding; the encoder output is
then folded 64x (sherpa's length cap must cover the search steps plus the emitted tokens), and digital silence returns empty
text (peak under 1e-4, the Python and JS runners' test), through SherpaFrontend, SherpaEncoder and BeamDecoder.
Then quantize_onnx.py <out_dir> <dir2> cq, and sherpa_pack.py <dir> <pack_dir> to lay the pack out."""
import json
import os
import sys

import numpy as np
import onnx
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quantize_onnx import share_tied_embedding  # noqa: E402
from sherpa_beam import BeamDecoder  # noqa: E402
from shruti_torch import Decoder, Encoder, Frontend, load, native_pair  # noqa: E402

FOLD = 16


class SherpaFrontend(torch.nn.Module):
    """The silence guard's first half, the same test as the Python and JS runners: peak under 1e-4 gives zero features."""

    def __init__(self, fe):
        super().__init__()
        self.fe = fe

    def forward(self, wav):
        return self.fe(wav) * (wav.abs().max() >= 1e-4).to(wav.dtype)


class SherpaEncoder(torch.nn.Module):
    def __init__(self, enc, fold=FOLD, guard=False):
        super().__init__()
        self.enc, self.fold, self.guard = enc, fold, guard

    def forward(self, feats, feats_len):  # feats_len: sherpa passes feats.shape[1]; the last frame is padding
        n = (torch.zeros(1, dtype=torch.int64) + feats.shape[1] - 1) + 0 * feats_len.to(torch.int64)
        c = self.enc(feats, n)
        if self.guard:  # zero features give an all-zero output, which BeamDecoder answers with eos at once
            c = c * (feats.abs().max() > 0).to(c.dtype)
        return c.reshape(1, -1, c.shape[-1] // self.fold)


class SherpaDecoder(torch.nn.Module):
    def __init__(self, dec, width):
        super().__init__()
        self.dec, self.width = dec, width

    def forward(self, tokens, cross, seq_len, *state):
        state = list(state)
        state[0] = state[0] + 0 * seq_len.to(torch.int64)  # sherpa's seq_len is not needed (the state has the position)
        return self.dec(tokens.to(torch.int64), cross.reshape(1, -1, self.width), *state)


def main():
    ckpt, out_dir, clip = sys.argv[1:4]
    beam = int(sys.argv[sys.argv.index("--beam") + 1]) if "--beam" in sys.argv else 0
    fold = 64 if beam else FOLD
    os.makedirs(out_dir, exist_ok=True)
    w, cfg, run = load(ckpt)
    fe, enc = Frontend(cfg).eval(), Encoder(w, cfg).eval()
    if beam:
        fe = SherpaFrontend(fe).eval()
    width = cfg["dec_layers"] * (cfg["dim"] // cfg["v_dim"]) * (cfg["qk_dim"] + cfg["v_dim"])
    if beam:
        dec = sdec = BeamDecoder(Decoder(w, cfg, swap_bos_eos=False, mask_tags=False).eval(), cfg, width, fold, K=beam)
        init_state = dec.init_state
    else:
        dec = Decoder(w, cfg, swap_bos_eos=True, mask_tags=True, first_tags=native_pair(cfg)).eval()
        sdec = SherpaDecoder(dec, width).eval()
        init_state = lambda: dec.init_state(1)  # noqa: E731
    senc = SherpaEncoder(enc, fold, guard=bool(beam)).eval()
    sdec = sdec.eval()
    wav, _ = sf.read(clip, dtype="float32")
    names = dec.state_names()
    with torch.no_grad():
        audio = torch.from_numpy(wav[None])
        feats = fe(audio)
        flen = torch.tensor([feats.shape[1]], dtype=torch.int32)
        cross = senc(feats, flen)
        st = list(init_state().values())
        one = torch.ones(1, dtype=torch.int32)
        for tok in (1, cfg["text"] + 7, 500, 600):  # a few calls so the traced state is mid-search, cache non-empty
            out = sdec(torch.tensor([[tok]], dtype=torch.int32), cross, one, *st)
            st = list(out[1:])
    kw = dict(opset_version=18, dynamo=False, do_constant_folding=True)
    torch.onnx.export(fe, (audio,), f"{out_dir}/frontend.onnx", input_names=["audio"], output_names=["features"],
                      dynamic_axes={"audio": {1: "samples"}, "features": {1: "frames"}}, **kw)
    torch.onnx.export(senc, (feats, flen), f"{out_dir}/encoder.onnx", input_names=["features", "features_len"],
                      output_names=["encoder_out"], dynamic_axes={"features": {1: "frames"}, "encoder_out": {1: "rows"}},
                      **kw)
    dyn = {"cross": {1: "rows"}}
    for s in names:
        if s[0] in "kv" and s[1:].isdigit():
            dyn[s], dyn[f"new_{s}"] = {2: "past"}, {2: "past_new"}
        if s in ("seqs", "best_seq", "out_seq"):
            dyn[s], dyn[f"new_{s}"] = {1: "calls"}, {1: "calls_new"}
    torch.onnx.export(sdec, (torch.tensor([[500]], dtype=torch.int32), cross, one, *st), f"{out_dir}/decoder.onnx",
                      input_names=["tokens", "cross", "seq_len"] + names,
                      output_names=["logits"] + [f"new_{s}" for s in names], dynamic_axes=dyn, **kw)
    share_tied_embedding(f"{out_dir}/decoder.onnx")
    # initial state for uncached_decode, and the token list
    init = {k: v.numpy() for k, v in init_state().items()}
    np.savez(f"{out_dir}/init_state.npz", **init)
    import sentencepiece as spm
    base = spm.SentencePieceProcessor(model_file=os.path.join(HERE, "../whistle/needle_src/needle/model/tokenizer.model"))
    vocab = [base.id_to_piece(i) for i in range(base.get_piece_size())] + list(run["pieces"])
    json.dump({"config": cfg, "vocab": vocab + [f"<|{lang}|>" for lang in cfg["languages"]], "sherpa_ids": True,
               "control": [i for i in range(base.get_piece_size()) if base.is_control(i) or base.is_unknown(i)]},
              open(f"{out_dir}/shruti.json", "w", encoding="utf-8"), ensure_ascii=False)
    for f in ("frontend", "encoder", "decoder"):
        print(f, f"{os.path.getsize(f'{out_dir}/{f}.onnx') / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
