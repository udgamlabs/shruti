# Shruti Maithili

Maithili and English in one 29.7 MB file that runs on your device. File: `shruti-maithili.cact`. Maithili comes out in Devanagari, and it keeps English words in English letters, as here: आब packet बना कऽ online बेचबाक विचार कऽ रहल छी. Built on Cactus Whistle via Shruti Hindi; Apache-2.0 for this model. Weights and model card: [huggingface.co/udgamlabs/shruti-maithili](https://huggingface.co/udgamlabs/shruti-maithili).

**We make no comparative claim for this model.** IndicConformer Maithili is ahead on IndicVoices valid (39.3 vs 42.5) and on DiarBench's native reference (40.2 vs 51.2). With English counted in English letters, the 54-turn test shows no lead either way (45.7 vs 46.9, an interval that includes 0). What it offers is one small file that works offline. No Maithili reader has reviewed this output yet.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Maithili in its on-device engine on a Mac. Measured 3 November 2026. Size is the file as published.

| Model | File size | IndicVoices valid | DiarBench, English in English | DiarBench, native |
|---|---|---|---|---|
| **Shruti Maithili** | 29.7 MB | 42.5 | 45.7 | 51.2 |
| [IndicConformer Maithili](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-mai-onnx) (int8) | 137.7 MB | 39.3 | 46.9 | 40.2 |

IndicVoices valid is IndicVoices' own held-out split: its speakers are kept out of training and its sentences blocked, but it is the same source as our training data, so it flatters us. DiarBench is real conversation from outside it: 54 turns from 15 conversations, 727 reference words, 109 of them English, in 30 turns. The "English in English" column scores against references that write English words in English (Latin script), and English words count only if written in English letters; the native column scores against references that write those English words in Devanagari. Paired 95% intervals are Shruti Maithili minus the other model, in word error points: negative means fewer errors for Shruti Maithili, and an interval that includes 0 shows no lead either way.

- **IndicConformer Maithili:** IndicVoices valid +1.01 to +5.73 by clip; DiarBench native +6.67 to +14.77 by clip, +6.62 to +14.84 by conversation, +6.12 to +15.56 with script ignored; DiarBench English in English -6.29 to +3.88 by clip, -5.54 to +3.23 by conversation, -2.81 to +7.26 with script ignored.
- **English words in English letters:** 46.8% of DiarBench's 109 English words, on 54 turns. IndicConformer Maithili writes none.
- **English:** LibriSpeech test-clean 6.04 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Maithili speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 9.1x faster than real time, 64 MB peak memory (20 clips).

## Where it falls short

- **IndicConformer Maithili is ahead or level on every test.** IndicVoices valid 42.5 vs 39.3 (+1.01 to +5.73), DiarBench native 51.2 vs 40.2 (+6.67 to +14.77), DiarBench English in English 45.7 vs 46.9 (-6.29 to +3.88). Character errors on DiarBench 31.26% vs 15.02%.
- **A small test.** 54 turns from 15 conversations, 30 of them with English, 109 English words. It can't say much either way, including whether the two are level.
- **Unmeasured rivals.** `rockerritesh/whisper-tiny-maithili` (151.1 MB, about 44 MB as q8_0) and others exist. We have not measured them.
- **English words in Devanagari:** about 34 of DiarBench's 109 by automatic alignment (shastri as शास्त्री). DiarBench also glosses some local names in Latin, so a correct native spelling can score as an error.
- **Browser:** tested on 5 lines only, played into the page on one phone's Chrome with the language set. All 5 came out in Devanagari, and 3 kept some English letters. Not tested with a live microphone. On the other Shruti languages where we measured it, plain English loses a little accuracy with the language set.
- **No Maithili reader has reviewed this output yet.**
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Maithili).
- **Not tested on Maithili:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
