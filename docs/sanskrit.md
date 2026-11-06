# Shruti Sanskrit

Spoken Sanskrit and English in one 29.7 MB file that runs on your device (`shruti-sanskrit.cact`). Sanskrit comes out in Devanagari
and English words stay in English. Built on Cactus Whistle via Shruti Hindi; Apache-2.0. Weights and model card:
[huggingface.co/udgamlabs/shruti-sanskrit](https://huggingface.co/udgamlabs/shruti-sanskrit).

**We make no comparative claim for this model.** IndicConformer Sanskrit is more accurate on IndicVoices' held-out set and ahead on
conversation against the native reference. Chant, verse and recitation were not tested.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Sanskrit in its on-device engine on a Mac.
IndicVoices valid has 200 clips, DiarBench 55 turns. Measured 3 November 2026; Su-śrotā on 6 November 2026. Size is the file as
published (whisper.cpp q8_0 for Whisper).

| Model | File size | IndicVoices valid | DiarBench, English in English (conversation, held out) | DiarBench, native reference |
|---|---|---|---|---|
| **Shruti Sanskrit** | 29.7 MB | 30.9 | 32.3 | 38.1 |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 43.5 MB | 617.4 | 522.9 | 528.1 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 81.8 MB | 440.6 | 367.7 | 374.7 |
| [IndicConformer Sanskrit](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-sa-onnx) (int8) | 137.7 MB | 24.4 | 36.6 | 33.4 |
| [Su-śrotā](https://huggingface.co/gnumanth/sushrota-sanskrit-asr-onnx) (int8 ONNX) | 187.5 MB | 38.7 | 43.8 | 41.8 |
| [Whisper small](https://huggingface.co/openai/whisper-small) | 264.5 MB | 443.3 | 367.4 | 369.4 |

IndicVoices valid is IndicVoices' own held-out split: its speakers are kept out of training and its sentences blocked, but it is the
same source as our training data, so it flatters us. DiarBench is real conversation: 55 turns from 21 conversations, 68 English words
in 637, too few to say much about mixed speech. Its "English in English" references write English words in English (Latin script), and
English words count only if written in English letters; its native references write those English words in Devanagari.

- **IndicConformer Sanskrit:** ahead on IndicVoices valid (paired 95% interval of the difference +4.29 to +9.01 points). DiarBench native
  -0.18 to +9.65 by clip, +0.98 to +8.46 by conversation. DiarBench English in English -11.44 to +2.77, which includes 0: no lead either way.
- **Su-śrotā:** word errors on IndicVoices valid -11.00 to -4.54, DiarBench native -9.20 to +1.69. It writes words joined more often, and
  word error counts each join as errors. On characters it is level or ahead (IndicVoices valid 11.62% vs 11.75%, DiarBench native 12.13% vs
  15.90%). We claim no lead over it.
- **English:** LibriSpeech test-clean 5.93 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Sanskrit speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 12.0x faster than real time, 64 MB peak memory (20
clips).

## Where it falls short

- **Not the most accurate.** See above. Character errors against IndicConformer Sanskrit: IndicVoices valid 11.75% vs 6.10%, DiarBench native
  15.90% vs 8.65%.
- **About a third of words are wrong** on both tests. Sandhi spacing overstates this for word error, but expect to correct a lot.
- **A thin test.** No independent read set; DiarBench is 55 turns. No Sanskrit reader has reviewed this output yet.
- **Chant, verse and recitation were not tested.**
- **English words in Devanagari:** about 16 of DiarBench's 68 English words come out in Devanagari (egypt as इजिप्ट्).
- **Browser:** tested on five lines only, played into the page on one phone's Chrome. Not tested with a live microphone.
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Sanskrit).
- **Not tested on Sanskrit:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
