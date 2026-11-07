# How the ONNX builds are made

These scripts write the released models' 4-bit weights as standard ONNX operators. Order: `dequant_w4.py` puts the shipped weights back in a float checkpoint, `export_onnx.py` writes the graphs, `quantize_onnx.py` makes the builds, `sherpa_export.py` and `sherpa_pack.py` lay out the sherpa-onnx pack with the search from `sherpa_beam.py`. `shruti_torch.py` is the PyTorch port they use. Built on Cactus Whistle (Apache-2.0); `dequant_w4.py` needs its `needle` package.

The builds themselves and every result: [udgamlabs/Shruti-ONNX](https://huggingface.co/udgamlabs/Shruti-ONNX).
