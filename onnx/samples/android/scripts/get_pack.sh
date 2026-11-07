#!/bin/sh
# Bundles one language's pack in the APK: scripts/get_pack.sh ur
# Each pack adds about 53 MB to the app. Language codes: docs/languages.md.
# Set SHRUTI_PACK_DIR to copy from a local folder instead of downloading (the folder holding <code>/sherpa-onnx/).
set -eu
cd "$(dirname "$0")/.."
code=${1:?usage: get_pack.sh <language code>}
out="app/src/main/assets/models/$code"
mkdir -p "$out"
for f in preprocess.onnx encode.onnx uncached_decode.onnx cached_decode.onnx tokens.txt; do
  if [ -n "${SHRUTI_PACK_DIR:-}" ]; then
    cp "$SHRUTI_PACK_DIR/$code/sherpa-onnx/$f" "$out/$f"
  else
    curl -fL --retry 3 -o "$out/$f" "https://huggingface.co/udgamlabs/Shruti-ONNX/resolve/main/$code/sherpa-onnx/$f"
  fi
done
echo "$out is ready. Build the app."
