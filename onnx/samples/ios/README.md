# Shruti in Swift, for iOS

**Not built or run on an iPhone or the iOS Simulator.** `ShrutiTranscriber.swift` has been compiled and run on a Mac only, against sherpa-onnx
2.0.1's desktop libraries and a real Shruti pack (`check_on_mac/`). That checks the Swift code and the pack. It says nothing about iOS builds,
memory or speed on an iPhone.

To try it on iOS:

1. Build sherpa-onnx's iOS framework with its own `build-ios.sh` from the [v2.0.1 source](https://github.com/k2-fsa/sherpa-onnx/tree/v2.0.1). The
   2.0.1 release has no ready-made iOS download.
2. Start from sherpa-onnx's `ios-swiftui` example, and add `SherpaOnnx.swift` from `swift-api-examples/` and `ShrutiTranscriber.swift` from here.
3. Add one language's `sherpa-onnx/` folder and `silero_vad.onnx` to the app's resources, and pass their paths to `ShrutiTranscriber`.

```swift
let asr = ShrutiTranscriber(pack: packFolder, vadModel: vadPath)
let pieces = asr.transcribe(samples16k)   // any length; long audio is split at pauses
```

If you only need Shruti on iOS and don't already use sherpa-onnx, the [needle engine](../../docs/integrate.md#the-engine-route) is the other route; it
also supports iOS.
