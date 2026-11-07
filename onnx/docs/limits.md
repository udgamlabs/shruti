# Where it falls short, and what to do

**9 of the 22 ONNX builds failed our test against the engine:** Hindi, Tamil, Kannada, Odia, Dogri, Telugu, Punjabi, Konkani and Sindhi. We publish them with the misses stated, and every number is in
[languages.md](languages.md). Odia also misses our character check.

Other numbers here were measured on Hindi unless a line says otherwise, on the builds as tested. Other languages can differ, so check
[languages.md](languages.md).

## Long audio: split it first

| Route | What happens | What you do |
|---|---|---|
| sherpa-onnx pack | Unsplit, a 60-second clip repeated a phrase after about 38 s (171 words against the engine's 147), and took 49 s and 420 MB | Split with a voice-activity detector (VAD) first. sherpa-onnx ships one, and our samples use it with pieces capped at 28 s |
| Python and JavaScript packages | Audio is cut into equal windows of at most 30 s, wherever the cut falls | Nothing needed, but a VAD first avoids cutting through a word |
| Needle engine | Handles long audio itself: 60 s in 9.2 s on the phone we tested | Nothing needed |

## Silence and quiet

The rebuilt packs return nothing on digital silence (all-zero audio). Real-world quiet is not digital silence, so use a VAD
for it, as the Android and C++ samples do. The samples also check for all-zero audio themselves, which costs nothing.

## Indian English on automatic detection

Each model detects between its own language and English. On Indian-accented English it often picks the wrong one: in Hindi,
word error on the Svarah test set was 29.4 with the language detected and 24.9 with English set.

| Route | How to set English |
|---|---|
| Python | `asr.transcribe(audio, language="en")` |
| JavaScript | `asr.transcribe(samples, { language: "en" })` |
| Needle engine | `model.transcribe(audio, language="en")`, or `--audio-language en` |
| sherpa-onnx pack | **Not possible.** Its Moonshine API has no language option. If your app is all-English speech on a phone, use the engine |

Set English only for speech that is entirely English. Leave it on detect for the model's own language and for mixed speech:
on the Hindi sets, detection scored the same as setting the language.

## Numbers as digits

Numbers come out as spoken words, not digits. If your app needs digits, convert them in your app after recognition.

## If the model repeats itself

Now and then the model gets stuck repeating a phrase: 117 of 9,536 test clips on the 4-bit build. The Python and JavaScript packages keep its output as it is by default, which matches the
engine and the sherpa-onnx pack. To remove everything after the first copy of a repeat, pass `loops="cut"` in Python or `{ loops: "cut" }` in JavaScript. That also removes any real words
after the repeat.

## Memory

| Build | Memory after loading (Mac) |
|---|---|
| `onnx/`, 4-bit | About 0.8 to 1 GB, because ONNX Runtime expands the weights when it loads |
| `onnx-int8/` | About 120 MB |
| `sherpa-onnx/` | About 160 to 200 MB; on the phone we tested about 190 MB per clip, at most 234 MB |

The 32 MB browser build is the smallest download but not the lightest to run. Browser memory is **not measured**, so test the
4-bit build on the low-end devices you care about, and offer `variant: "int8"` if it struggles.

## Other limits

- **On a phone, the original engine is faster and lighter:** about 1.5 s for a 10-second clip including loading, in 63 MB.
  The sherpa-onnx pack takes about 2.4 s after a 3.4 s load, in about 190 MB.
- **No word-by-word text in the sherpa-onnx pack:** the search runs inside the model, so text arrives when a piece of audio
  ends. The Android sample shows it phrase by phrase by splitting at pauses. The engine has a live mode.
- **Small differences between builds:** the same pack can differ by a letter or a word between devices and runtime builds. On
  50 clips, 37 were identical between the phone and the Mac. The needle engine does this too: the same file can write a
  slightly different word on a different chip.
- **Linux on x86:** on Hindi and Urdu, the 32 MB 4-bit build wrote identical text on a Mac and a Linux x86 machine (1,274 clips). The
  low-memory build and the sherpa-onnx pack did not: 83 of 200 Hindi and 29 of 74 Urdu conversation clips matched, and Linux was usually
  less accurate on conversation, on an AMD EPYC without VNNI; chips with VNNI are untested. On a Linux x86 server, use the 4-bit build if you can spare the memory. Every language:
  [RESULTS_LINUX_X86.md](https://huggingface.co/udgamlabs/Shruti-ONNX/blob/main/RESULTS_LINUX_X86.md).
- **Sindhi** comes out in Devanagari, not Perso-Arabic script.
- **Names it never heard** are spelt by sound, and far-away or speaker-played audio throws it. Speak close to the device.
- **Pure Hindi:** larger models are more accurate. Shruti's strength is mixed speech and size.
- **Scripts need fonts:** make sure your app or page can show the language's script. Older phones may lack fonts for some of
  them. We have not tested this across devices.
- **Not tested:** iPhone, Windows, phones other than the Galaxy F62, an Android app built on the engine, a pack bundled inside an Android app, and clips over 30 s beyond the 60-second clip above.
