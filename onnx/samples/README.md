# Shruti samples

Small working examples. The guide that goes with them: [`docs/integrate.md`](../docs/integrate.md).

| Folder | What it is | Run on |
|---|---|---|
| [`android/`](android/) | Kotlin app, sherpa-onnx 2.0.1, file and microphone, language setting, no network | Galaxy F62 |
| [`browser/`](browser/) | One page with a language picker, works offline after the first load, no build step | Mac, Chromium-based browser |
| [`python/`](python/) | `transcribe.py`, on the `shruti-onnx` package | Mac |
| [`node/`](node/) | `transcribe.mjs`, on the `shruti-onnx` package | Mac |
| [`cpp/`](cpp/) | `transcribe.cc`, on sherpa-onnx 2.0.1 | Mac |
| [`ios/`](ios/) | Swift code on sherpa-onnx's Swift API | Mac only. **Not run on an iPhone** |
| [`engine-c/`](engine-c/) | `transcribe.c`, on the Cactus engine's C library, whole clip or live | Mac; Galaxy F62 with the Linux arm64 library |

Windows: not run.
