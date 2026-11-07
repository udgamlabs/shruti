"""Shruti's float checkpoint with the shipped 4-bit weights put back in (the .cact's weights, dequantised).
Every CQ leaf (kernels, embeddings, mHC phi; needle quantize._is_quant_leaf) goes through needle's own cq_quantize at
4 bits, group 128, along its reduction axis, which reproduces the exported .cact weights; every
other leaf is rounded to FP16 as the .cact stores it. The ONNX export of this file is the shipped model's weights
without the engine's 8-bit activations.
Usage: python dequant_w4.py <full_model.safetensors> <out.safetensors>"""
import os
import sys

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../whistle/needle_src"))
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.numpy import load_file, save_file  # noqa: E402
from needle.model.quantize import cq_quantize  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
w = load_file(src)
with safe_open(src, framework="np") as h:
    meta = h.metadata()
out = {}
for k, v in w.items():
    leaf = k.split("/")[-1]
    if v.ndim >= 2 and (leaf in ("kernel", "embedding") or leaf.startswith("mhc_phi")):
        second_last = leaf == "kernel" or leaf.startswith("mhc_phi")
        x = np.swapaxes(v, -1, -2) if second_last else v
        q = np.asarray(cq_quantize(jnp.asarray(x, jnp.float32), 4, 128))
        out[k] = np.ascontiguousarray(np.swapaxes(q, -1, -2) if second_last else q, np.float32)
        print(f"CQ4 {k} {v.shape} rel err {np.linalg.norm(out[k] - v) / np.linalg.norm(v):.3f}", flush=True)
    else:
        out[k] = v.astype(np.float16).astype(np.float32)
save_file(out, dst, metadata=meta)
