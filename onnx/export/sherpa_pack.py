"""Lay out a sherpa-onnx Moonshine-v1 pack from sherpa_export.py graphs (float or quantize_onnx.py output).
Usage: python sherpa_pack.py <graphs_dir> <pack_dir>
Writes preprocess.onnx, encode.onnx, uncached_decode.onnx (built here), cached_decode.onnx and tokens.txt."""
import json
import os
import shutil
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    meta = json.load(open(os.path.join(src, "shruti.json"), encoding="utf-8"))
    cfg, vocab = meta["config"], meta["vocab"]
    V = len(vocab)
    for a, b in (("frontend", "preprocess"), ("encoder", "encode"), ("decoder", "cached_decode")):
        shutil.copy(os.path.join(src, f"{a}.onnx"), os.path.join(dst, f"{b}.onnx"))
    # uncached_decode: constant first logits (argmax = the 'de' tag, absent from tokens.txt) + the empty state
    init = np.load(os.path.join(src, "init_state.npz"))
    names = [i.name for i in onnx.load(os.path.join(src, "decoder.onnx")).graph.input][3:]
    first = np.zeros((1, 1, V), np.float32)
    first[0, 0, cfg["text"] + cfg["languages"].index("de")] = 1.0
    nodes, outs = [], []
    for name, arr in [("logits", first)] + [(f"new_{n}", init[n]) for n in names]:
        nodes.append(helper.make_node("Constant", [], [name], value=numpy_helper.from_array(arr, name + "_value")))
        dt = TensorProto.INT64 if arr.dtype == np.int64 else TensorProto.FLOAT
        outs.append(helper.make_tensor_value_info(name, dt, list(arr.shape)))
    ins = [helper.make_tensor_value_info("tokens", TensorProto.INT32, [1, 1]),
           helper.make_tensor_value_info("cross", TensorProto.FLOAT, [1, "rows", "width"]),
           helper.make_tensor_value_info("seq_len", TensorProto.INT32, [1])]
    g = helper.make_graph(nodes, "shruti_uncached", ins, outs)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], producer_name="udgamlabs")
    m.ir_version = onnx.load(os.path.join(src, "decoder.onnx")).ir_version
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(dst, "uncached_decode.onnx"))
    # tokens.txt in sherpa ids (needle bos 2 / eos 1 swapped); control pieces and language tags left out
    perm = list(range(V))
    perm[cfg["bos"]], perm[cfg["eos"]] = perm[cfg["eos"]], perm[cfg["bos"]]
    skip = set(meta.get("control", [])) | set(range(cfg["text"], V))
    lines = []
    for ext in range(V):
        i = perm[ext]
        if i in skip:
            continue
        sym = vocab[i]
        assert sym and not any(c.isspace() for c in sym), (i, sym)
        lines.append(f"{sym} {ext}")
    open(os.path.join(dst, "tokens.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    total = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
    print(f"pack {dst}: {len(lines)} tokens, {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
