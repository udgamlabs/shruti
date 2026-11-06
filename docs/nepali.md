# Shruti Nepali

Nepali and English in one 29.7 MB file that runs on your device. File: `shruti-nepali.cact`. Nepali comes out in Devanagari, and it keeps English words in English letters, as here: दिदी बफको मोमो एक plate र एउटा cold drink दिनु न. Built on Cactus Whistle via Shruti Hindi; Apache-2.0 for this model. Weights and model card: [huggingface.co/udgamlabs/shruti-nepali](https://huggingface.co/udgamlabs/shruti-nepali).

**We make no comparative claim for this model.** IndicConformer Nepali is well ahead on read Nepali (FLEURS 29.9 vs 46.0), on IndicVoices valid (18.4 vs 27.1) and on DiarBench's native reference (28.6 vs 40.5). **Set the language when you know it's Nepali.** Left to choose on its own, the engine sent 20 of 200 FLEURS clips wholly to English letters; with Nepali set, none did, and FLEURS moves from 46.0 to 42.2. That helps and closes part of the gap, not all of it. The try box on udgamlabs.com sets the picked language and so does our demo on Hugging Face. What it offers is one small file that works offline. No Nepali reader has reviewed this output yet.

**Set the language when you know it's Nepali:** `python examples/transcribe.py clip.wav --model nepali --language ne`. See below.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Nepali in its on-device engine on a Mac. Measured 3 November 2026. Size is the file as published (whisper.cpp q8_0 for Whisper).

| Model | File size | FLEURS (read) | IndicVoices valid | DiarBench, English in English | DiarBench, native |
|---|---|---|---|---|---|
| **Shruti Nepali** | 29.7 MB | 46.0 | 27.1 | 34.0 | 40.5 |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 43.5 MB | 270.5 | 488.6 | 308.4 | 310.5 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 81.8 MB | 200.4 | 371.4 | 287.2 | 288.3 |
| [IndicConformer Nepali](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-ne-onnx) (int8) | 137.7 MB | 29.9 | 18.4 | 37.4 | 28.6 |
| [Whisper small](https://huggingface.co/openai/whisper-small) | 264.5 MB | 126.7 | 226.5 | 238.0 | 237.6 |

FLEURS is read speech we did not train on. DiarBench is real conversation from outside our training data. IndicVoices valid is IndicVoices' own held-out split: its speakers are kept out of training and its sentences blocked, but it is the same source as our training data, so it flatters us. DiarBench has 45 turns from 19 conversations, 874 reference words, 118 of them English. The "English in English" column scores against references that write English words in English (Latin script), and English words count only if written in English letters; the native column scores against references that write those English words in Devanagari. Whisper's rates pass 100% because it adds words and often writes the wrong script, so they say little about Whisper at its best. Paired 95% intervals are Shruti Nepali minus the other model, in word error points: negative means fewer errors for Shruti Nepali, and an interval that includes 0 shows no lead either way.

- **IndicConformer Nepali:** FLEURS +13.60 to +19.38 by clip; IndicVoices valid +6.72 to +11.13 by clip; DiarBench native +8.07 to +16.62 by clip, +7.98 to +16.63 by conversation, +7.99 to +16.55 with script ignored; DiarBench English in English -8.67 to +2.66 by clip, -8.83 to +2.37 by conversation, -6.30 to +4.54 with script ignored.
- **English words in English letters:** 55.9% of DiarBench's 118 English words, on 45 turns. Stock IndicConformer Nepali writes none.
- **English:** LibriSpeech test-clean 6.01 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Nepali speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 9.7x faster than real time, 64 MB peak memory (20 clips).

## Where it falls short

- **Read Nepali is far behind.** FLEURS 46.0 vs IndicConformer Nepali's 29.9 (+13.60 to +19.38); 42.2 with the language set. Character errors 24.94% vs 10.55%.
- **The automatic language choice misfires.** 20 of 200 FLEURS clips came out wholly in English letters with the engine's own choice. With Nepali set, none of the 200 did, and the set moves from 46.0 to 42.2, still behind. On the other Shruti languages where we measured it, plain English loses a little accuracy with the language set.
- **The other tests favour IndicConformer Nepali too.** IndicVoices valid 27.1 vs 18.4 (+6.72 to +11.13), DiarBench native 40.5 vs 28.6 (+8.07 to +16.62). DiarBench English in English shows no lead either way (34.0 vs 37.4; -8.67 to +2.66), on 45 turns and 118 English words, too few to say much.
- **A code-mix rival exists that we have not measured.** `ailabpvtltd/indicconformer-nepali-english-codemix-asr` (513.5 MB, CC BY-NC 4.0) is built for exactly this mix and reports WER 19.9 on its own set, not comparable until measured. So what we say about IndicConformer Nepali writing no English holds for the stock model.
- **English words in Devanagari:** about 22 of DiarBench's 118 by automatic alignment. DiarBench also glosses some local names in Latin, so a correct native spelling can score as an error.
- **Browser:** tested on 5 lines only, played into the page on one phone's Chrome with the language set. All 5 came out in Devanagari, and 4 kept some English letters. Not tested with a live microphone. On the other Shruti languages where we measured it, plain English loses a little accuracy with the language set.
- **No Nepali reader has reviewed this output yet.**
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Nepali).
- **Not tested on Nepali:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
