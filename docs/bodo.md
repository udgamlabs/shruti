# Shruti Bodo

Bodo and English in one 29.7 MB file. On Bodo conversations, about 27% fewer errors than IndicConformer Bodo, a model almost five times its size. File: `shruti-bodo.cact`. Bodo comes out in Devanagari, and about half of the English words stay in English letters: आंनि फिसाया गुवाहाटीयाव फरायो दिनै hostel fee होनांगौ. Built on Cactus Whistle via Shruti Hindi; Apache-2.0 for this model. Weights and model card: [huggingface.co/udgamlabs/shruti-bodo](https://huggingface.co/udgamlabs/shruti-bodo).

**The scope matters.** On DiarBench (held out, all 112 usable turns from 25 conversations, 97 of them with English words), counting English words only in English letters, its word error rate is 39.9 against IndicConformer Bodo's 55.0, and that holds by conversation and with script ignored. Against the native reference on the same turns IndicConformer Bodo is ahead (46.9 vs 51.5), and on IndicVoices valid too (28.6 vs 34.5). It is the one rival we measured. No Bodo reader has reviewed this output yet.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Bodo in its on-device engine on a Mac. Measured 3 November 2026. Size is the file as published.

| Model | File size | IndicVoices valid | DiarBench, English in English | DiarBench, native |
|---|---|---|---|---|
| **Shruti Bodo** | 29.7 MB | 34.5 | 39.9 | 51.5 |
| [IndicConformer Bodo](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-brx-onnx) (int8) | 137.7 MB | 28.6 | 55.0 | 46.9 |

IndicVoices valid is IndicVoices' own held-out split: its speakers are kept out of training and its sentences blocked, but it is the same source as our training data, so it flatters us. DiarBench is real conversation from outside it: 112 turns from 25 conversations, 1,592 reference words, 409 of them English, in 97 turns. The "English in English" column scores against references that write English words in English (Latin script), and English words count only if written in English letters; the native column scores against references that write those English words in Devanagari. Paired 95% intervals are Shruti Bodo minus the other model, in word error points: negative means fewer errors for Shruti Bodo, and an interval that includes 0 shows no lead either way.

- **IndicConformer Bodo:** IndicVoices valid +3.47 to +8.51 by clip; DiarBench native +1.77 to +7.65 by clip, +1.28 to +8.03 by conversation, +2.62 to +8.40 with script ignored; DiarBench English in English -19.42 to -10.63 by clip, -19.89 to -10.54 by conversation, -17.42 to -8.43 with script ignored.
- **English words in English letters:** 50.4% of DiarBench's 409 English words, on 112 turns. IndicConformer Bodo writes none.
- **English:** LibriSpeech test-clean 5.99 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Bodo speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 10.9x faster than real time, 64 MB peak memory (20 clips).

## Where it falls short

- **One rival measured.** Whisper, SraVaani-1.0, Dolphin and Vosk do not list Bodo. Other Bodo models exist that we have not measured.
- **The native reference shows the gap.** IndicConformer Bodo is ahead on DiarBench native (46.9 vs 51.5) and on IndicVoices valid (28.6 vs 34.5). Character errors: DiarBench 28.99% vs 14.58%, IndicVoices valid 14.75% vs 8.34%.
- **No independent read test.** IndicVoices valid shares a source with our training data, and no independent Bodo read set was available.
- **English words in Devanagari:** about 102 of DiarBench's 409 by automatic alignment (france as फ्रेन्स). DiarBench also glosses some local names in Latin, so a correct native spelling can score as an error.
- **Browser:** tested on 4 test clips only, played into the page on one phone's Chrome with the language set. All 4 came out in Devanagari. These are test clips read by an AI voice, not real speech. Not tested with a live microphone. On the other Shruti languages where we measured it, plain English loses a little accuracy with the language set.
- **No Bodo reader has reviewed this output yet.**
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Bodo).
- **Not tested on Bodo:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
