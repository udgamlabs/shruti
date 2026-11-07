#!/bin/sh
# Downloads the two things the app builds against and checks their SHA-256:
#   app/libs/sherpa-onnx-2.0.1.aar          sherpa-onnx's Android library, pinned at 2.0.1
#   app/src/main/assets/silero_vad.onnx     the voice-activity detector (Silero VAD, MIT)
set -eu
cd "$(dirname "$0")/.."

check() { # file sha256
  if command -v sha256sum >/dev/null; then echo "$2  $1" | sha256sum -c -; else echo "$2  $1" | shasum -a 256 -c -; fi
}
get() { # url file sha256
  if [ -f "$2" ] && check "$2" "$3" >/dev/null 2>&1; then echo "have $2"; return; fi
  mkdir -p "$(dirname "$2")"
  curl -fL --retry 3 -o "$2" "$1"
  check "$2" "$3"
}

REL=https://github.com/k2-fsa/sherpa-onnx/releases/download
get "$REL/v2.0.1/sherpa-onnx-2.0.1.aar" app/libs/sherpa-onnx-2.0.1.aar \
  a8d09d1d32f3eb8520482c6d3d43fae674681c5b050385967c70a3c2c9bf0ee7
get "$REL/asr-models/silero_vad.onnx" app/src/main/assets/silero_vad.onnx \
  9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6
