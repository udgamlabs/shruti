#!/bin/sh
# Compiles the Swift code for macOS against sherpa-onnx 2.0.1's desktop libraries, then runs it. This checks the Swift against
# the real API and the real pack. It does not test an iPhone.
#   sh ../cpp/get_sherpa.sh   (once, from samples/cpp)
#   sh check_on_mac.sh <pack folder> clip.wav
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
CPP="$HERE/../../cpp"; SHERPA="$CPP/sherpa-onnx"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
TAG=https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v2.0.1/swift-api-examples
curl -fsSL -o "$W/SherpaOnnx.swift" "$TAG/SherpaOnnx.swift"
curl -fsSL -o "$W/Bridge.h" "$TAG/SherpaOnnx-Bridging-Header.h"
cp "$HERE/../ShrutiTranscriber.swift" "$HERE/main.swift" "$W/"
(cd "$W" && swiftc -O -lc++ -I "$SHERPA/include" -L "$SHERPA/lib" -lsherpa-onnx-c-api -lonnxruntime \
  -Xlinker -rpath -Xlinker "$SHERPA/lib" -import-objc-header Bridge.h -o main SherpaOnnx.swift ShrutiTranscriber.swift main.swift)
"$W/main" "$1" "$CPP/silero_vad.onnx" "$2"
