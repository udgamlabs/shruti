#!/bin/sh
# Downloads sherpa-onnx 2.0.1's libraries for this machine into ./sherpa-onnx, and silero_vad.onnx, and checks SHA-256.
# macOS on Apple silicon and Linux x86_64 are listed here. Other platforms: https://github.com/k2-fsa/sherpa-onnx/releases/tag/v2.0.1
set -eu
cd "$(dirname "$0")"

check() { if command -v sha256sum >/dev/null; then echo "$2  $1" | sha256sum -c -; else echo "$2  $1" | shasum -a 256 -c -; fi; }

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) name=osx-arm64; sha=5e874d3c9b3d6b8c71f7c09e3a9fb9ae846c3009129d302b398d1f7aedb69f84 ;;
  Linux-x86_64) name=linux-x64; sha=2e823376987c3b6960d50726b6560155e2ad92da0b8b3886325773225ec6bdec ;;
  *) echo "add your platform's download here"; exit 1 ;;
esac

REL=https://github.com/k2-fsa/sherpa-onnx/releases/download
file="sherpa-onnx-v2.0.1-$name-shared.tar.bz2"  # the -shared-lib tarballs have no headers
curl -fL --retry 3 -o "$file" "$REL/v2.0.1/$file"
check "$file" "$sha"
rm -rf sherpa-onnx && mkdir sherpa-onnx
tar xjf "$file" -C sherpa-onnx --strip-components=1
rm "$file"

curl -fL --retry 3 -o silero_vad.onnx "$REL/asr-models/silero_vad.onnx"
check silero_vad.onnx 9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6
