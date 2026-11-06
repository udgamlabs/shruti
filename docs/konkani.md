# Shruti Konkani

Konkani and English in one 29.7 MB file. On Konkani conversations, about 15% fewer errors than IndicConformer Konkani, a model almost five times its size. File: `shruti-konkani.cact`. Konkani comes out in Devanagari, and about half of the English words stay in English letters: बांगडे आनी सुंगटां fridge भितर दवरल्यात. Built on Cactus Whistle via Shruti Hindi; Apache-2.0 for this model. Weights and model card: [huggingface.co/udgamlabs/shruti-konkani](https://huggingface.co/udgamlabs/shruti-konkani).

**The scope matters.** On DiarBench (held out, all 90 usable turns from 23 conversations, 71 of them with English words), counting English words only in English letters, its word error rate is 40.0 against IndicConformer Konkani's 47.0, and that holds by conversation and with script ignored. Against the native reference on the same turns IndicConformer Konkani is well ahead (37.3 vs 47.3), and on IndicVoices valid too (33.4 vs 41.4). It is the one rival we measured. It writes Devanagari Konkani, not Romi or Kannada script. No Konkani reader has reviewed this output yet.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Konkani in its on-device engine on a Mac. Measured 3 November 2026. Size is the file as published.

| Model | File size | IndicVoices valid | DiarBench, English in English | DiarBench, native |
|---|---|---|---|---|
| **Shruti Konkani** | 29.7 MB | 41.4 | 40.0 | 47.3 |
| [IndicConformer Konkani](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-kok-onnx) (int8) | 137.7 MB | 33.4 | 47.0 | 37.3 |

IndicVoices valid is IndicVoices' own held-out split: its speakers are kept out of training and its sentences blocked, but it is the same source as our training data, so it flatters us. DiarBench is real conversation from outside it: 90 turns from 23 conversations, 1,457 reference words, 278 of them English, in 71 turns. The "English in English" column scores against references that write English words in English (Latin script), and English words count only if written in English letters; the native column scores against references that write those English words in Devanagari. Paired 95% intervals are Shruti Konkani minus the other model, in word error points: negative means fewer errors for Shruti Konkani, and an interval that includes 0 shows no lead either way.

- **IndicConformer Konkani:** IndicVoices valid +6.10 to +10.00 by clip; DiarBench native +6.41 to +13.71 by clip, +6.60 to +13.63 by conversation, +6.69 to +12.53 with script ignored; DiarBench English in English -10.64 to -3.57 by clip, -10.86 to -3.50 by conversation, -6.96 to -0.90 with script ignored.
- **English words in English letters:** 49.6% of DiarBench's 278 English words, on 90 turns. IndicConformer Konkani writes none.
- **English:** LibriSpeech test-clean 6.0 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Konkani speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 11.7x faster than real time, 64 MB peak memory (20 clips).

## Where it falls short

- **Devanagari only.** Goan Konkani is widely written in Romi too, and this model does not write it. `Reubencf/konkani-whisper-stt` targets Romi; we have not measured it.
- **The native reference shows the gap.** IndicConformer Konkani is ahead on DiarBench native (37.3 vs 47.3; +6.41 to +13.71) and on IndicVoices valid (33.4 vs 41.4). Character errors: DiarBench 25.38% vs 11.58%, IndicVoices valid 22.77% vs 10.1%.
- **One rival measured.** No independent read test was available either: IndicVoices valid shares a source with our training data.
- **English words in Devanagari:** about 77 of DiarBench's 278 by automatic alignment (so as सो). DiarBench also glosses some local names in Latin, so a correct native spelling can score as an error.
- **Browser:** tested on 4 lines only, played into the page on one phone's Chrome with the language set. All 4 came out in Devanagari, and 2 kept some English letters. Not tested with a live microphone. On the other Shruti languages where we measured it, plain English loses a little accuracy with the language set.
- **No Konkani reader has reviewed this output yet.**
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Konkani).
- **Not tested on Konkani:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
