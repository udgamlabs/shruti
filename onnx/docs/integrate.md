# Add Shruti to your app

Shruti is offline speech-to-text: one small model for each of 22 languages, each model also covering English and the two
mixed. Nothing is sent to a server. Pick the route that fits what you are building, then copy its quickstart. All 22 are on every route, but the ONNX and sherpa-onnx builds of
9 languages failed our test against the engine, and we publish them with the misses stated: see [languages.md](languages.md).

## Choose your route

| You are building | Use | Download per language | Can you set the language? | Tested |
|---|---|---|---|---|
| A Python app or a server | [`shruti-onnx` for Python](#python) | 32 MB, or 53 MB for low memory | Yes | Mac |
| A web page | [`shruti-onnx` for the browser](#browser) | 32 MB | Yes | Mac, Chromium-based browser |
| A Node app | [`shruti-onnx` for Node](#node) | 32 MB | Yes | Mac |
| An Android app | [sherpa-onnx pack](#android) | 53 MB | No, always detected | Galaxy F62 |
| A desktop app in C or C++ | [sherpa-onnx pack](#c-and-c) | 53 MB | No, always detected | Mac |
| An iOS app | [sherpa-onnx pack](#ios) | 53 MB | No, always detected | **Not run on an iPhone** |
| A faster, lighter phone app, live word-by-word text, or setting the language on a phone | [Cactus engine](#the-engine-route) | 29.5 to 29.7 MB | Yes | Mac, Galaxy F62 |

Windows: **not run** on any route.

**Which one on a phone?** On the phone we tested, the needle engine is faster and lighter than the sherpa-onnx pack: a
10-second clip takes about 1.5 s including loading, in 63 MB, against about 2.4 s after a 3.4 s load, in about 190 MB. It
is also harder to add to an app, because you link its C library yourself. Choose sherpa-onnx if your app already uses
it or you want a ready-made Android library; choose the engine if speed and memory matter most.

Languages and download links: [languages.md](languages.md). What to watch for: [limits.md](limits.md).

## Python

```bash
pip install shruti-onnx
```

```python
from shruti_onnx import Shruti

asr = Shruti.load("urdu")                                # a word or a code ("ur"); downloads once, then works offline
print(asr.transcribe("clip.wav")["text"])                # detects Urdu or English
print(asr.transcribe("clip.wav", language="en")["text"]) # all-English speech, especially with an Indian accent
```

- `transcribe` also takes 16 kHz mono float32 samples. WAV files work out of the box; `pip install soundfile` adds other
  formats.
- `Shruti.load("urdu", variant="int8")` is the low-memory build: a 53 MB download and about 120 MB of memory, instead of
  a 32 MB download and about 1 GB.
- `Shruti.load("urdu", model="/my/folder")` uses your own copy instead of downloading. `LANGUAGES` lists the codes.

Command line: `shruti-onnx clip.wav --lang urdu`. Runnable sample: [`samples/python/`](../samples/python/).

## Browser

```bash
npm i shruti-onnx onnxruntime-web
```

```js
import * as ort from "onnxruntime-web";
import { createShruti } from "shruti-onnx";

const asr = await createShruti({ ort, language: "urdu" });    // a word or a code; downloads the model
const { text } = await asr.transcribe(samples16k);            // Float32Array, 16 kHz mono
await asr.transcribe(samples16k, { language: "en" });         // all-English speech
```

No build step: [`samples/browser/`](../samples/browser/) is one page with a language picker, a file button and a
microphone button, and it works offline after the first load. Serve the folder over `https` or `localhost` (the microphone
and offline support both need that):

```bash
cd onnx/samples/browser && python3 -m http.server 8000
```

## Node

The same package and calls as the browser. Set `ort.env.wasm.numThreads = 1`. `baseUrl` can be a local folder path.

```bash
cd onnx/samples/node && npm install
node transcribe.mjs clip.wav urdu
```

## Android

[`samples/android/`](../samples/android/) is a small Kotlin app: pick a language in its settings, open a WAV file or use the
microphone. It has no `INTERNET` permission, so audio and text cannot leave the phone.

It uses sherpa-onnx 2.0.1's Android library. Copy `ShrutiEngine.kt` into your own app:

```kotlin
val asr = ShrutiEngine(assets, packFolder, packInAssets = false)   // off the main thread: this takes a few seconds
asr.transcribe(samples16k) { text -> show(text) }                  // any length; long audio is split at pauses
```

To build the sample:

```bash
cd onnx/samples/android
sh scripts/get_deps.sh      # sherpa-onnx 2.0.1 and the voice-activity detector, with checksums
sh scripts/get_pack.sh ur   # bundles the Urdu pack in the app; each pack adds about 53 MB
./gradlew assembleDebug
```

We ran this app on the Galaxy F62 with packs pushed to the phone's folder. **A pack bundled inside the app, as `get_pack.sh` does, has not been run**, and neither has anyone speaking into its
microphone. To use the path we ran, skip `get_pack.sh` and push the pack: see [`samples/android/`](../samples/android/).

Use 2 threads, the fastest on the phone we tested. Keep the screen on or use a foreground service for long jobs: with the
screen off, Android slowed loading from 3.2 s to 9.4 s.

## C and C++

```bash
cd onnx/samples/cpp
sh get_sherpa.sh            # sherpa-onnx 2.0.1 and the voice-activity detector, with checksums
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/transcribe /path/to/ur/sherpa-onnx clip.wav silero_vad.onnx
```

`get_sherpa.sh` lists macOS on Apple silicon and Linux x86_64; this C++ sample was run on the Mac only. On Linux x86 the pack's text can
differ and is usually less accurate on conversation: see [limits.md](limits.md). Input is a 16 kHz mono WAV file.

## iOS

**Not built or run on an iPhone or the iOS Simulator.** [`samples/ios/ShrutiTranscriber.swift`](../samples/ios/) is the same
logic as the Android sample, in Swift, through sherpa-onnx's Swift API. We compiled it and ran it on a Mac against the
real pack, which checks the Swift code but says nothing about iOS itself.

sherpa-onnx 2.0.1 does not publish a ready-made iOS framework: build it with sherpa-onnx's own `build-ios.sh`, and start from
its `ios-swiftui` example.

## The engine route

Shruti's original files run in the Cactus engine, on Mac, Linux, Windows, Android, iOS and WebAssembly. This is the
route with 22 `.cact` files, one per language.

```bash
pip install cactus-compute
```

```python
from huggingface_hub import hf_hub_download   # installed with cactus-compute
from cactus import Whistle

model = Whistle(hf_hub_download("udgamlabs/Shruti", "shruti.cact"))
print(model.transcribe("clip.wav"))                    # the language is detected
print(model.transcribe("english.wav", language="en"))  # all-English speech
```

The `cactus-compute` Python package sends anonymous usage counts by default; set `CACTUS_TELEMETRY=0` to turn that off. The
command-line engine and the WebAssembly build make no network calls.

From the command line, the `cactus` binary ([Cactus-Compute/cactus](https://huggingface.co/Cactus-Compute/cactus/tree/main/releases/4.1.0)) needs no Python:

```bash
cactus --model shruti.cact --audio clip.wav
cactus --model shruti.cact --audio talk.wav --audio-stream        # live, word by word
cactus --model shruti.cact --audio english.wav --audio-language en
```

In a browser, the public repo's `examples/browser/` is one page with no build step that runs the engine's WebAssembly build:

```bash
git clone --depth 1 https://github.com/udgamlabs/shruti && cd shruti/examples/browser
python3 -m http.server 8000     # then open http://localhost:8000/?model=tamil
```

On an Android phone, over USB, the engine's Android binary shows it working before you write any code:

```bash
curl -fLO https://huggingface.co/Cactus-Compute/cactus/resolve/4e822c9036be2eb0e47ef556df32a818bb1572ef/releases/4.1.0/android-arm64/cactus
curl -fLO https://huggingface.co/udgamlabs/shruti-tamil/resolve/main/shruti-tamil.cact
adb push cactus shruti-tamil.cact clip.wav /data/local/tmp/
adb shell chmod +x /data/local/tmp/cactus
adb shell "cd /data/local/tmp && ./cactus --model shruti-tamil.cact --audio clip.wav --threads 2"
```

In an app, link the engine's C library (`cactus.h`: `cactus_load`, `cactus_transcribe`, `cactus_stream_transcribe_process`)
and ship the `.cact` file with the app. [`samples/engine-c/transcribe.c`](../samples/engine-c/) shows those calls in one short file; we ran it on a Mac and, with the Linux
arm64 library, on the phone. On Android the library comes from the same repo's `android-arm64/` folder and needs the NDK's C++ runtime; we have not yet built and run an app
that way. This is the extra work the sherpa-onnx route saves you. Use 2 threads on phones with two fast cores. Details: the
[Shruti README](https://github.com/udgamlabs/shruti).

## Licence

Apache-2.0, like Shruti and Cactus Whistle. Keep the NOTICE file. sherpa-onnx (Apache-2.0), ONNX Runtime (MIT) and Silero VAD
(MIT) have their own licences.
