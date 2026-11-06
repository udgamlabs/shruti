# Shruti Sindhi

Sindhi, written in Devanagari, and English in one 29.7 MB file that runs on your device. File: `shruti-sindhi.cact`. Sindhi comes out in Devanagari, and it keeps English words in English letters, as here: छोकिरे जो biodata ऐं photo मोकिलियो. Built on Cactus Whistle via Shruti Hindi; Apache-2.0 for this model. Weights and model card: [huggingface.co/udgamlabs/shruti-sindhi](https://huggingface.co/udgamlabs/shruti-sindhi).

**It writes Sindhi in Devanagari, not Perso-Arabic script,** so readers of Perso-Arabic Sindhi can't use its output. **We make no comparative claim for this model.** IndicVoices valid is level with IndicConformer Sindhi (33.0 vs 33.1), on a set that shares a source with our training data, and the conversation test is only 27 turns. What it offers is one small file that works offline. No Sindhi reader has reviewed this output yet.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Sindhi in its on-device engine on a Mac. Measured 3 November 2026. Size is the file as published.

| Model | File size | IndicVoices valid | DiarBench, English in English | DiarBench, native |
|---|---|---|---|---|
| **Shruti Sindhi** | 29.7 MB | 33.0 | 34.2 | 38.0 |
| [IndicConformer Sindhi](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-sd-onnx) (int8) | 137.7 MB | 33.1 | 38.7 | 35.8 |

Whisper is not in this table: it hardly writes Devanagari on these sets, so its error rates would measure script, not accuracy. IndicVoices valid is IndicVoices' own held-out split: its speakers are kept out of training and its sentences blocked, but it is the same source as our training data, so it flatters us. DiarBench is real conversation from outside it: 27 turns from 17 conversations, 971 reference words, 51 of them English, in 12 turns. The "English in English" column scores against references that write English words in English (Latin script), and English words count only if written in English letters; the native column scores against references that write those English words in Devanagari. Paired 95% intervals are Shruti Sindhi minus the other model, in word error points: negative means fewer errors for Shruti Sindhi, and an interval that includes 0 shows no lead either way.

- **IndicConformer Sindhi:** IndicVoices valid -2.45 to +2.39 by clip; DiarBench native -1.53 to +6.04 by clip, -1.43 to +4.91 by conversation, +1.99 to +8.99 with script ignored; DiarBench English in English -8.37 to +0.10 by clip, -8.46 to -0.36 by conversation, -4.55 to +4.21 with script ignored.
- **English:** LibriSpeech test-clean 6.05 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Sindhi speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 10.8x faster than real time, 64 MB peak memory (20 clips).

## Where it falls short

- **Devanagari only.** Readers of Perso-Arabic Sindhi can't use the output. FLEURS Sindhi is in Arabic script and was left out for this reason, so there is no independent read test.
- **A small test.** DiarBench Sindhi is 27 turns from 17 conversations with 51 English words, too few to say much.
- **The level result is on the flattering set.** IndicVoices valid (-2.45 to +2.39) shares a source with our training data. Character errors there 19.9% vs 12.41%, and on DiarBench 21.64% vs 14.33%.
- **English words in Devanagari:** about 7 of DiarBench's 51 by automatic alignment. DiarBench also glosses some local names in Latin, so a correct native spelling can score as an error.
- **Browser:** tested on 4 lines only, played into the page on one phone's Chrome with the language set. All 4 came out in Devanagari, and 2 kept some English letters. Not tested with a live microphone. On the other Shruti languages where we measured it, plain English loses a little accuracy with the language set.
- **No Sindhi reader has reviewed this output yet.**
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Sindhi).
- **Not tested on Sindhi:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
