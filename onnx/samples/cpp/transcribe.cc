// Transcribe a 16 kHz mono WAV file with a Shruti pack in sherpa-onnx.
//
//   ./transcribe <pack folder> clip.wav [silero_vad.onnx]
//
// There is no language option: sherpa-onnx's Moonshine API detects between the pack's language and English.
// Up to 30 s is decoded whole. Longer audio needs silero_vad.onnx, which splits it at pauses first, because the pack starts
// repeating itself on long unsplit audio.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/cxx-api.h"

namespace sx = sherpa_onnx::cxx;

constexpr int32_t kRate = 16000;
constexpr int32_t kWindow = 512;  // the detector takes exactly this many samples at a time
constexpr size_t kOnePass = 30 * kRate;

static std::string Decode(const sx::OfflineRecognizer &recognizer, const float *x, size_t n) {
  // The same silence test as the Python and JavaScript packages. Real-world quiet is not silent: use the VAD for that.
  if (std::all_of(x, x + n, [](float v) { return std::fabs(v) < 1e-4f; })) return "";
  sx::OfflineStream stream = recognizer.CreateStream();
  stream.AcceptWaveform(kRate, x, static_cast<int32_t>(n));
  recognizer.Decode(&stream);
  return recognizer.GetResult(&stream).text;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage: transcribe <pack folder> clip.wav [silero_vad.onnx]\n";
    return 2;
  }
  const std::string pack = argv[1];
  const sx::Wave wave = sx::ReadWave(argv[2]);
  if (wave.samples.empty() || wave.sample_rate != kRate) {
    std::cerr << "need a 16 kHz mono WAV: ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav\n";
    return 1;
  }

  sx::OfflineRecognizerConfig config;
  config.model_config.moonshine.preprocessor = pack + "/preprocess.onnx";
  config.model_config.moonshine.encoder = pack + "/encode.onnx";
  config.model_config.moonshine.uncached_decoder = pack + "/uncached_decode.onnx";
  config.model_config.moonshine.cached_decoder = pack + "/cached_decode.onnx";
  config.model_config.tokens = pack + "/tokens.txt";
  config.model_config.num_threads = 2;
  const sx::OfflineRecognizer recognizer = sx::OfflineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "could not load the pack from " << pack << "\n";
    return 1;
  }

  const std::vector<float> &x = wave.samples;
  if (x.size() <= kOnePass) {
    std::cout << Decode(recognizer, x.data(), x.size()) << "\n";
    return 0;
  }
  if (argc < 4) {
    std::cerr << "this clip is over 30 s: pass silero_vad.onnx so it can be split first\n";
    return 2;
  }

  sx::VadModelConfig vad_config;
  vad_config.silero_vad.model = argv[3];
  vad_config.silero_vad.window_size = kWindow;
  vad_config.silero_vad.min_silence_duration = 0.25f;
  vad_config.silero_vad.max_speech_duration = 28.0f;  // stay under 30 s per piece
  vad_config.sample_rate = kRate;
  const sx::VoiceActivityDetector vad = sx::VoiceActivityDetector::Create(vad_config, 60.0f);

  const auto drain = [&] {
    while (!vad.IsEmpty()) {
      const sx::SpeechSegment piece = vad.Front();
      const std::string text = Decode(recognizer, piece.samples.data(), piece.samples.size());
      if (!text.empty()) std::cout << text << "\n";
      vad.Pop();
    }
  };
  for (size_t i = 0; i + kWindow <= x.size(); i += kWindow) {
    vad.AcceptWaveform(x.data() + i, kWindow);
    drain();
  }
  vad.Flush();
  drain();
  return 0;
}
