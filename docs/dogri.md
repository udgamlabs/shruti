# Shruti Dogri

Dogri and English in one 29.7 MB file that runs on your device. File: `shruti-dogri.cact`. Dogri comes out in Devanagari, and it keeps English words in English letters, as here: कटड़े आह्‌ली बस हून चलने आह्‌ली ऐ ticket लेई लैओ. Built on Cactus Whistle via Shruti Hindi; Apache-2.0 for this model. Weights and model card: [huggingface.co/udgamlabs/shruti-dogri](https://huggingface.co/udgamlabs/shruti-dogri).

**We make no comparative claim for this model.** IndicConformer Dogri is ahead on IndicVoices valid, the one sizeable test (38.5 vs 43.5), and the Dogri conversation test is only 24 turns, too few to say anything either way. What it offers is one small file that works offline. No Dogri reader has reviewed this output yet.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Dogri in its on-device engine on a Mac. Measured 3 November 2026. Size is the file as published.

| Model | File size | IndicVoices valid | DiarBench, English in English | DiarBench, native |
|---|---|---|---|---|
| **Shruti Dogri** | 29.7 MB | 43.5 | 34.5 | 37.0 |
| [IndicConformer Dogri](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-doi-onnx) (int8) | 137.7 MB | 38.5 | 39.7 | 37.4 |

IndicVoices valid is IndicVoices' own held-out split: its speakers are kept out of training and its sentences blocked, but it is the same source as our training data, so it flatters us. DiarBench is real conversation from outside it: 24 turns from 17 conversations, 479 reference words, 40 of them English, in 16 turns. The "English in English" column scores against references that write English words in English (Latin script), and English words count only if written in English letters; the native column scores against references that write those English words in Devanagari. Paired 95% intervals are Shruti Dogri minus the other model, in word error points: negative means fewer errors for Shruti Dogri, and an interval that includes 0 shows no lead either way.

- **IndicConformer Dogri:** IndicVoices valid +2.58 to +7.27 by clip; DiarBench native -5.31 to +5.32 by clip, -6.15 to +5.64 by conversation, -1.74 to +10.86 with script ignored; DiarBench English in English -10.99 to +1.78 by clip, -12.13 to +1.99 by conversation, -7.30 to +6.09 with script ignored.
- **English:** LibriSpeech test-clean 6.1 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Dogri speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 9.5x faster than real time, 64 MB peak memory (20 clips).

## Where it falls short

- **The test can't support a claim.** DiarBench Dogri is 24 turns from 17 conversations, 479 words, 40 English words. No independent read set was available.
- **IndicConformer Dogri is ahead** on IndicVoices valid (38.5 vs 43.5; +2.58 to +7.27). Character errors: 26.59% vs 17.91% there, 25.64% vs 16.98% on DiarBench.
- **One rival measured.** Whisper does not list Dogri. Other Dogri models exist that we have not measured.
- **English words in Devanagari:** about 12 of DiarBench's 40 by automatic alignment. DiarBench also glosses some local names in Latin, so a correct native spelling can score as an error.
- **Browser:** tested on 4 lines only, played into the page on one phone's Chrome with the language set. All 4 came out in Devanagari, and 3 kept some English letters. Not tested with a live microphone. On the other Shruti languages where we measured it, plain English loses a little accuracy with the language set.
- **No Dogri reader has reviewed this output yet.**
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Dogri).
- **Not tested on Dogri:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
