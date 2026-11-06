# Shruti Santali

Santali in Ol Chiki and English in one 29.7 MB file that runs on your device. File: `shruti-santali.cact`. Santali comes out in Ol Chiki, and some English words stay in English letters, as here: ᱤᱱᱟᱹ ᱵᱚᱭᱦᱟ ᱡᱟᱢᱥᱮᱫᱽᱯᱩᱨ ᱨᱮ ᱢᱤᱫᱴᱟᱝ company ᱨᱮ ᱠᱟᱹᱢᱤᱭᱮᱫᱟᱭ. Built on Cactus Whistle via Shruti Hindi; Apache-2.0 for this model. Weights and model card: [huggingface.co/udgamlabs/shruti-santali](https://huggingface.co/udgamlabs/shruti-santali).

**On our two tests we found no clear difference from IndicConformer Santali,** a model almost five times its size. On DiarBench (99 turns, native reference) 33.6 vs 34.2, on IndicVoices valid 40.3 vs 42.6, and both intervals include 0. IndicConformer Santali has fewer character errors on both (11.7% vs 15.88%, 13.3% vs 16.88%). Don't read this as a lead: the size is the difference. Its English test is thin, 63 English words. No Santali reader has reviewed this output yet.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Santali in its on-device engine on a Mac. Measured 3 November 2026. Size is the file as published.

| Model | File size | IndicVoices valid | DiarBench, English in English | DiarBench, native |
|---|---|---|---|---|
| **Shruti Santali** | 29.7 MB | 40.3 | 33.0 | 33.6 |
| [IndicConformer Santali](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-sat-onnx) (int8) | 137.7 MB | 42.6 | 35.4 | 34.2 |

IndicVoices valid is IndicVoices' own held-out split: its speakers are kept out of training and its sentences blocked, but it is the same source as our training data, so it flatters us. DiarBench is real conversation from outside it: 99 turns from 28 conversations, 2,897 reference words, 63 of them English, in 34 turns. The "English in English" column scores against references that write English words in English (Latin script), and English words count only if written in English letters; the native column scores against references that write those English words in Ol Chiki. Paired 95% intervals are Shruti Santali minus the other model, in word error points: negative means fewer errors for Shruti Santali, and an interval that includes 0 shows no lead either way.

- **IndicConformer Santali:** IndicVoices valid -5.29 to +0.52 by clip; DiarBench native -3.16 to +2.03 by clip, -2.67 to +1.61 by conversation; DiarBench English in English -4.94 to +0.25 by clip, -4.60 to +0.10 by conversation.
- **English words in English letters:** 36.5% of DiarBench's 63 English words, on 99 turns. IndicConformer Santali writes none.
- **English:** LibriSpeech test-clean 5.98 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Santali speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 12.6x faster than real time, 64 MB peak memory (20 clips).

## Where it falls short

- **No lead over the one rival measured.** DiarBench native -3.16 to +2.03, DiarBench English in English -4.94 to +0.25, IndicVoices valid -5.29 to +0.52. Character errors are higher: 15.88% vs 11.7% and 16.88% vs 13.3%.
- **Ol Chiki only.** It does not write Santali in Devanagari, Bengali, Odia or Latin script, and no script-ignored check exists for Ol Chiki.
- **A thin English test.** 63 English words in 2,897, of which 36.5% come out in English letters.
- **English words in Ol Chiki:** about 30 of DiarBench's 63 by automatic alignment (doctor as ᱰᱟᱠᱛᱟᱨ). DiarBench also glosses some local names in Latin, so a correct native spelling can score as an error.
- **Browser:** tested on 4 test clips only, played into the page on one phone's Chrome with the language set. All 4 came out in Ol Chiki. These are test clips read by an AI voice, not real speech. Not tested with a live microphone. On the other Shruti languages where we measured it, plain English loses a little accuracy with the language set.
- **No Santali reader has reviewed this output yet.**
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Santali).
- **Not tested on Santali:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
