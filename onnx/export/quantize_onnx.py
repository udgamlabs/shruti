"""Smaller builds of Shruti's float ONNX graphs (export_onnx.py output). Every build keeps the same inputs and outputs.
Usage: python quantize_onnx.py <fp32_dir> <out_dir> <mode> [options]
  fp16          weights stored as FP16, cast back to FP32 when the session loads (compute stays FP32)
  int8          ONNX Runtime dynamic quantisation: int8 per-channel weights, MatMul + Gather
  int4 [--block 32] [--sym]   ONNX Runtime MatMulNBits / GatherBlockQuantized (the int4 "q4" path transformers.js uses)
  cq8                         phone pack: the cq weights as int8 matrices at run time (low memory, ~1.7x the file)
  cq [--table-bits 4]         Cactus Quants, the shipped .cact's own 4-bit weight format, written with standard ONNX ops:
      per 128-weight group a Walsh-Hadamard rotation, an FP16 length and 4-bit codebook indices (needle export.py).
      Matrix weights are dequantised by a constant subgraph (ONNX Runtime folds it when the session loads); the n-gram
      memory tables stay packed and only the rows a token fetches are dequantised. --table-bits 2 stores the tables at
      2 bits (the Shruti v2 size test's free 24.8 MB map).
"""
import os
import re
import shutil
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
CQ_NAME = re.compile(r"(_kernel|wqkv|\.phi|stem_out|stem_pw_\d|\.ck|\.cv|(^|\.)emb(_T)?|\.tables|key_proj|value_proj)$")
GROUP = 128


def consumers(m, name):
    return [(n, [i for i, x in enumerate(n.input) if x == name]) for n in m.graph.node if name in n.input]


# ---------------- Cactus Quants (needle export._cq_pack), numpy ----------------
_cb = np.load(os.path.join(HERE, "cq_codebooks.npz"))
H = _cb["H"].astype(np.float32)


def cq_codes(rows, bits):
    """rows (R, D), D % 128 == 0 -> (idx (R, D) uint8, norm (R, D/128) fp16), exactly as needle's exporter."""
    cb = _cb[f"cb{bits}"].astype(np.float32)
    R, D = rows.shape
    g = rows.reshape(R, D // GROUP, GROUP).astype(np.float32)
    rot = g @ H
    norm = np.sqrt((rot ** 2).sum(-1, keepdims=True))
    unit = rot / np.maximum(norm, 1e-12)
    flat = unit.reshape(-1)
    pos = np.clip(np.searchsorted(cb, flat), 1, len(cb) - 1)
    left, right = cb[pos - 1], cb[pos]
    idx = np.where(np.abs(flat - left) <= np.abs(flat - right), pos - 1, pos).astype(np.uint8)
    return idx.reshape(R, D), norm[..., 0].astype(np.float16)


def pack(idx, bits):
    """(R, D) codes -> (R, D*bits/8) uint8, code j of a byte in bits [j*bits, (j+1)*bits) (low first)."""
    per = 8 // bits
    R, D = idx.shape
    g = idx.reshape(R, D // per, per).astype(np.uint16)
    return (g * (1 << (bits * np.arange(per)))).sum(-1).astype(np.uint8)


def cq_deq_np(idx, norm, bits):
    cb = _cb[f"cb{bits}"].astype(np.float32)
    R, D = idx.shape
    return ((cb[idx].reshape(R, D // GROUP, GROUP) * norm.astype(np.float32)[..., None]) @ H).reshape(R, D)


class Builder:
    def __init__(self, m, prefix):
        self.m, self.prefix, self.k = m, prefix, 0
        self.consts = {}

    def const(self, name, arr):
        if name not in self.consts:
            self.m.graph.initializer.append(numpy_helper.from_array(arr, name))
            self.consts[name] = name
        return name

    def node(self, op, ins, **attrs):
        self.k += 1
        out = f"{self.prefix}_{op}_{self.k}"
        self.m.graph.node.append(helper.make_node(op, ins, [out], name=out, **attrs))
        return out

    def lut(self, bits):
        """(256, 8/bits) codebook values of every byte: entry [b, j] = codebook[code j of b], code j in bits
        [j*bits, (j+1)*bits). One Gather on the byte replaces unpacking (no int64 copies of the codes at load)."""
        per = 8 // bits
        b = np.arange(256)[:, None]
        codes = (b >> (bits * np.arange(per))[None, :]) & ((1 << bits) - 1)
        return self.const(f"cq_lut{bits}", _cb[f"cb{bits}"].astype(np.float32)[codes])

    def dequant(self, packed, norm16, bits, flat_shape, group_shape):
        """packed uint8 (..., D*bits/8), norm fp16 (..., D/128) -> float (..., D) = (codebook[codes] * norm) @ H."""
        p = self.node("Cast", [packed], to=TensorProto.INT32)
        v = self.node("Gather", [self.lut(bits), p], axis=0)  # (..., D*bits/8, 8/bits)
        v = self.node("Reshape", [v, group_shape])
        nrm = self.node("Cast", [norm16], to=TensorProto.FLOAT)
        nrm = self.node("Unsqueeze", [nrm, self.const("cq_ax_last", np.array([-1], np.int64))])
        v = self.node("MatMul", [self.node("Mul", [v, nrm]), self.const("cq_H", H)])
        return self.node("Reshape", [v, flat_shape])


def to_cq(m, prefix, table_bits=4, report=None, materialize=False):
    """materialize: matrices become plain float initializers holding their 4-bit values (for int8 quantisation next);
    gathered tables stay packed either way."""
    inits = {i.name: i for i in m.graph.initializer}
    kept = set()
    b = Builder(m, prefix)
    max_err = 0.0
    for name, init in list(inits.items()):
        if not CQ_NAME.search(name) or len(init.dims) != 2:
            continue
        w = numpy_helper.to_array(init)
        uses = consumers(m, name)
        kinds = {n.op_type for n, _ in uses}
        if kinds == {"MatMul"} and all(pos == [1] for _, pos in uses):
            rows, layout = w.T, "matmul"  # x @ W, W (in, out): Cactus quantises W^T row by row along `in`
        elif kinds <= {"Gather", "Gemm"}:
            rows, layout = w, "rows"  # embedding / tables: one row per token or slot
        else:
            print("  skip (unexpected use)", name, kinds)
            continue
        if rows.shape[1] % GROUP:
            print("  skip (not a multiple of 128)", name, rows.shape)
            continue
        bits = table_bits if name.endswith(".tables") else 4
        idx, norm = cq_codes(rows, bits)
        deq = cq_deq_np(idx, norm, bits)
        max_err = max(max_err, float(np.linalg.norm(deq - rows) / np.linalg.norm(rows)))
        R, D = rows.shape
        packed = pack(idx, bits)
        m.graph.initializer.remove(init)
        if materialize and kinds != {"Gather"}:
            m.graph.initializer.append(numpy_helper.from_array(
                np.ascontiguousarray(deq.T if layout == "matmul" else deq, np.float32), name))
            kept.add(name)
            continue
        pk = b.const(f"{name}.cq{bits}", packed)
        nm = b.const(f"{name}.cqnorm", norm)
        if kinds == {"Gather"}:  # fetch packed rows, then dequantise only those
            for n, _ in uses:
                ids = n.input[1]
                out = n.output[0]
                gp = b.node("Gather", [pk, ids], axis=0)
                gn = b.node("Gather", [nm, ids], axis=0)
                sh = b.node("Shape", [ids])
                tail_c = b.node("Concat", [sh, b.const(f"cq_tail_D{D}", np.array([D], np.int64))], axis=0)
                tail_g = b.node("Concat", [sh, b.const(f"cq_tail_G{D}", np.array([D // GROUP, GROUP], np.int64))],
                                axis=0)
                v = b.dequant(gp, gn, bits, tail_c, tail_g)
                m.graph.node.remove(n)
                m.graph.node.append(helper.make_node("Identity", [v], [out], name=out + "_cq"))
        else:  # whole matrix, a constant subgraph folded at load
            v = b.dequant(pk, nm, bits, b.const(f"shape_{R}_{D}", np.array([R, D], np.int64)),
                          b.const(f"shape_{R}_{D // GROUP}_{GROUP}", np.array([R, D // GROUP, GROUP], np.int64)))
            if layout == "matmul":
                m.graph.node.append(helper.make_node("Transpose", [v], [name], perm=[1, 0], name=name + "_cqT"))
            else:
                m.graph.node.append(helper.make_node("Identity", [v], [name], name=name + "_cqI"))
        if report is not None:
            report.append((name, R, D, bits, packed.nbytes + norm.nbytes))
    print(f"  CQ max relative weight error {max_err:.3f}")
    to_fp16_storage(m, skip=lambda n: n.startswith("cq_") or ".cq" in n or n in kept)
    return topo_sort(m)


def to_fp16_storage(m, skip=lambda n: False, min_size=64):
    for init in list(m.graph.initializer):
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < min_size or skip(init.name):
            continue
        a = numpy_helper.to_array(init)
        m.graph.initializer.remove(init)
        m.graph.initializer.append(numpy_helper.from_array(a.astype(np.float16), init.name + ".fp16"))
        m.graph.node.append(helper.make_node("Cast", [init.name + ".fp16"], [init.name], to=TensorProto.FLOAT,
                                             name=init.name + "_cast"))
    return topo_sort(m)


def share_tied_embedding(path):
    """The exporter writes F.linear(h, emb) as MatMul(h, emb^T) with a second, transposed copy of the 18 MB table.
    Rewrite it as Gemm(h, emb, transB=1) and drop the copy."""
    m = onnx.load(path)
    inits = {i.name: i for i in m.graph.initializer}
    arrs = {n: numpy_helper.to_array(i) for n, i in inits.items() if len(i.dims) == 2 and np.prod(i.dims) > 1e6}
    for node in list(m.graph.node):
        if node.op_type != "MatMul" or node.input[1] not in arrs:
            continue
        src = next((n for n, a in arrs.items() if a.shape == arrs[node.input[1]].shape[::-1]
                    and np.array_equal(a.T, arrs[node.input[1]])), None)
        if src is None:
            continue
        gemm = helper.make_node("Gemm", [node.input[0], src], list(node.output), transB=1, name=node.name + "_gemm")
        idx = list(m.graph.node).index(node)
        m.graph.node.remove(node)
        m.graph.node.insert(idx, gemm)
        m.graph.initializer.remove(inits[node.input[1]])
        print(f"  {os.path.basename(path)}: MatMul with a transposed copy of {src} -> Gemm(transB=1)")
    onnx.save(m, path)


def untie_gemm(m):
    """The tied output layer is Gemm(h, emb, transB=1), which the int8 and int4 quantisers do not take (and the int8
    one breaks when the same table also feeds a Gather): make it a MatMul on its own transposed copy."""
    for n in list(m.graph.node):
        if n.op_type == "Gemm" and any(at.name == "transB" and at.i == 1 for at in n.attribute):
            w = next(i for i in m.graph.initializer if i.name == n.input[1])
            m.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(w).T.copy(), w.name + "_T"))
            mm = helper.make_node("MatMul", [n.input[0], w.name + "_T"], list(n.output), name=n.name + "_mm")
            m.graph.node.insert(list(m.graph.node).index(n), mm)
            m.graph.node.remove(n)
    return m


def topo_sort(m):
    known = {i.name for i in m.graph.input} | {i.name for i in m.graph.initializer} | {""}
    nodes, out = list(m.graph.node), []
    while nodes:
        rest = []
        for n in nodes:
            if all(x in known for x in n.input):
                out.append(n)
                known.update(n.output)
            else:
                rest.append(n)
        if len(rest) == len(nodes):
            raise RuntimeError(f"cycle or missing input: {rest[0].name} {list(rest[0].input)}")
        nodes = rest
    del m.graph.node[:]
    m.graph.node.extend(out)
    return m


def main():
    a = sys.argv[1:]
    src, dst, mode = a[0], a[1], a[2]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    os.makedirs(dst, exist_ok=True)
    for f in ("frontend.onnx", "shruti.json", "init_state.npz"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    for f in ("encoder.onnx", "decoder.onnx"):
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if mode == "fp16":
            onnx.save(to_fp16_storage(onnx.load(s)), d)
        elif mode == "int8":
            from onnxruntime.quantization import QuantType, quantize_dynamic
            tmp = d + ".untied.onnx"
            onnx.save(untie_gemm(onnx.load(s)), tmp)
            quantize_dynamic(tmp, d, op_types_to_quantize=["MatMul", "Gather"], per_channel=True,
                             weight_type=QuantType.QInt8)
            os.remove(tmp)
        elif mode == "int4":
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
            m = untie_gemm(onnx.load(s))
            inits = {i.name: i for i in m.graph.initializer}
            big = lambda x: x in inits and len(inits[x].dims) == 2 and inits[x].data_type == TensorProto.FLOAT \
                and np.prod(inits[x].dims) >= 16384  # noqa: E731
            include = [n.name for n in m.graph.node if (n.op_type == "MatMul" and big(n.input[1]))
                       or (n.op_type == "Gather" and big(n.input[0]))]
            q = MatMulNBitsQuantizer(m, bits=4, block_size=int(opt("--block", 32)), is_symmetric="--sym" in a,
                                     nodes_to_include=include, op_types_to_quantize=("MatMul",),
                                     quant_axes=(("MatMul", 0), ("Gather", 1)))
            q.process()
            m = to_fp16_storage(q.model.model)
            onnx.save(m, d)
        elif mode == "cq":
            rep = []
            onnx.save(to_cq(onnx.load(s), f.split(".")[0], int(opt("--table-bits", 4)), rep), d)
        elif mode == "cq8":
            # phone pack: the same 4-bit values, held as int8 matrices at run time (no float copy in memory); the
            # tables stay 4-bit packed; activations quantised on the fly (int8 MatMul, like the engine's 8-bit path)
            from onnxruntime.quantization import QuantType, quantize_dynamic
            m = to_cq(untie_gemm(onnx.load(s)), f.split(".")[0], int(opt("--table-bits", 4)), materialize=True)
            inits = {i.name: i for i in m.graph.initializer}
            big = [n.name for n in m.graph.node if n.op_type == "MatMul" and n.input[1] in inits
                   and np.prod(inits[n.input[1]].dims) >= 16384 and not n.input[1].startswith("cq_")]
            tmp = d + ".cq.onnx"
            onnx.save(m, tmp)
            quantize_dynamic(tmp, d, nodes_to_quantize=big, op_types_to_quantize=["MatMul"], per_channel=True,
                             weight_type=QuantType.QInt8)
            os.remove(tmp)
        else:
            raise SystemExit(f"unknown mode {mode}")
        print(f"{mode} {f}: {os.path.getsize(d) / 1e6:.2f} MB", flush=True)
    total = sum(os.path.getsize(os.path.join(dst, f)) for f in ("frontend.onnx", "encoder.onnx", "decoder.onnx"))
    print(f"{mode} total {total / 1e6:.2f} MB ({total} bytes)")


if __name__ == "__main__":
    main()
