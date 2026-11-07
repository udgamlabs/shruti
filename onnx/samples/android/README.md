# Shruti on Android (Kotlin, sherpa-onnx 2.0.1)

Pick a language, then open a WAV file or record from the microphone. The app has no `INTERNET` permission, so audio and text
cannot leave the phone. `ShrutiEngine.kt` is the part to copy into your own app.

## Build

You need Android Studio or the Android SDK, and Java 17.

```bash
sh scripts/get_deps.sh      # sherpa-onnx 2.0.1 and silero_vad.onnx, with checksums
sh scripts/get_pack.sh ur   # bundles a language in the app; each adds about 53 MB
./gradlew assembleDebug
```

To try a language without bundling it, push its pack instead and pick it in the app. This is the path we ran, on a Galaxy F62. **A pack bundled in the app has not been run.**

```bash
adb push ur/sherpa-onnx/. /sdcard/Android/data/com.udgamlabs.shruti.sample/files/models/ur/
```

## What's in it

| File | What it does |
|---|---|
| `ShrutiEngine.kt` | Loads a pack; `decode` for one piece of speech, `transcribe` for any length; `SpeechSplitter` cuts audio at pauses |
| `MainActivity.kt` | The screen: language setting, file picker, microphone. The microphone is read on its own thread, so text appears phrase by phrase |
| `Wav.kt` | Reads WAV files. For other formats use Android's `MediaCodec` |

## For your own app

- The app bundles or finds packs on the device. If yours downloads them, that one step needs the `INTERNET` permission and transcription still
  does not.
- `abiFilters` keeps only `arm64-v8a`: the app is 33 MB before any language pack. Add other processors if you need them.
- There is no language setting beyond choosing the pack: sherpa-onnx's Moonshine API always detects between the pack's language and English.
- Use 2 threads. Keep the screen on, or use a foreground service for long jobs: Android slows the CPU when the screen is off.
- Debug builds have two test hooks, `bench_dir` and `mic_test_wav`. Release builds do not run them.
