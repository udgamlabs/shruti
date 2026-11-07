// Shruti in Swift, through sherpa-onnx's Swift API (SherpaOnnx.swift, from sherpa-onnx v2.0.1's swift-api-examples folder).
//
// Compiled and run on macOS (see check_on_mac.sh). NOT yet built or run on an iPhone: see README.md.
//
// There is no language option: sherpa-onnx's Moonshine API detects between the pack's language and English.
import Foundation

final class ShrutiTranscriber {
  static let sampleRate = 16000
  private static let window = 512  // the detector takes exactly this many samples at a time

  private let recognizer: SherpaOnnxOfflineRecognizer
  private let vadModel: String

  /// - Parameters:
  ///   - pack: folder with preprocess.onnx, encode.onnx, uncached_decode.onnx, cached_decode.onnx and tokens.txt
  ///   - vadModel: path to silero_vad.onnx, used to split audio longer than 30 s
  init(pack: String, vadModel: String, threads: Int = 2) {
    let moonshine = sherpaOnnxOfflineMoonshineModelConfig(
      preprocessor: "\(pack)/preprocess.onnx",
      encoder: "\(pack)/encode.onnx",
      uncachedDecoder: "\(pack)/uncached_decode.onnx",
      cachedDecoder: "\(pack)/cached_decode.onnx")
    let model = sherpaOnnxOfflineModelConfig(
      tokens: "\(pack)/tokens.txt", numThreads: threads, moonshine: moonshine)
    var config = sherpaOnnxOfflineRecognizerConfig(
      featConfig: sherpaOnnxFeatureConfig(), modelConfig: model)
    recognizer = SherpaOnnxOfflineRecognizer(config: &config)
    self.vadModel = vadModel
  }

  /// Text for one piece of speech of up to 30 s, 16 kHz mono samples in [-1, 1].
  func decode(_ samples: [Float]) -> String {
    // The same silence test as the Python and JavaScript packages. Real-world quiet is not silent: use the VAD for that.
    if samples.allSatisfy({ abs($0) < 1e-4 }) { return "" }
    return recognizer.decode(samples: samples, sampleRate: Self.sampleRate).text
      .trimmingCharacters(in: .whitespaces)
  }

  /// Any length of audio. Up to 30 s is decoded whole; longer audio is split at pauses first, because the pack starts
  /// repeating itself on long unsplit audio.
  func transcribe(_ samples: [Float]) -> [String] {
    if samples.count <= 30 * Self.sampleRate { return [decode(samples)] }
    let silero = sherpaOnnxSileroVadModelConfig(
      model: vadModel, threshold: 0.5, minSilenceDuration: 0.25, minSpeechDuration: 0.25,
      windowSize: Self.window, maxSpeechDuration: 28)
    var config = sherpaOnnxVadModelConfig(sileroVad: silero, sampleRate: Int32(Self.sampleRate))
    let vad = SherpaOnnxVoiceActivityDetectorWrapper(config: &config, buffer_size_in_seconds: 60)
    var pieces: [String] = []
    let drain = {
      while !vad.isEmpty() {
        pieces.append(self.decode(vad.front().samples))
        vad.pop()
      }
    }
    var i = 0
    while i + Self.window <= samples.count {
      vad.acceptWaveform(samples: Array(samples[i..<(i + Self.window)]))
      drain()
      i += Self.window
    }
    vad.flush()
    drain()
    return pieces.filter { !$0.isEmpty }
  }
}
