# Shruti Punjabi

Punjabi and English in one 29.7 MB file that runs on your device. File: `shruti-punjabi.cact`. Punjabi comes out in Gurmukhi, and more than half of the English words stay in English letters: ਯਾਰੋ ਮੇਰਾ canada ਦਾ study visa ਆ ਗਿਆ ਹੈ. Built on Cactus Whistle via Shruti Hindi; Apache-2.0 for this model. Weights and model card: [huggingface.co/udgamlabs/shruti-punjabi](https://huggingface.co/udgamlabs/shruti-punjabi).

**On Punjabi conversations we found it level with IndicConformer Punjabi and SraVaani-1.0, not ahead.** On DiarBench (held out, 200 turns from 44 conversations), counting English words only in English letters: 32.6 vs 34.6 and 31.9, and both paired intervals include 0. It writes 58.7% of the English words in English letters, where those two write none. Against the native reference on the same turns they are well ahead (28.4 and 23.9 vs 38.3), and on read Punjabi too (FLEURS 16.9 and 15.1 vs 29.3). No Punjabi reader has reviewed this output yet.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Punjabi in its on-device engine on a Mac. Measured 2 November 2026. Size is the file as published (whisper.cpp q8_0 for Whisper).

| Model | File size | FLEURS (read) | Kathbath test-unknown (read) | DiarBench, English in English | DiarBench, native |
|---|---|---|---|---|---|
| **Shruti Punjabi** | 29.7 MB | 29.3 | 18.6 | 32.6 | 38.3 |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 43.5 MB | 311.5 | 337.8 | 468.7 | 467.7 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 81.8 MB | 201.6 | 207.9 | 275.9 | 275.3 |
| [Dolphin base](https://huggingface.co/DataoceanAI/dolphin-base) | 103.7 MB | 83.6 | 71.6 | 98.5 | 99.4 |
| [IndicConformer Punjabi](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-pa-onnx) (int8) | 137.7 MB | 16.9 | 13.7 | 34.6 | 28.4 |
| [Dolphin small](https://huggingface.co/DataoceanAI/dolphin-small) | 249.7 MB | 62.2 | 51.0 | 99.2 | 99.2 |
| [Whisper small](https://huggingface.co/openai/whisper-small) | 264.5 MB | 129.4 | 123.4 | 145.1 | 144.7 |
| [SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | 908.8 MB | 15.1 | 12.7 | 31.9 | 23.9 |

FLEURS is read speech we did not train on. Kathbath test-unknown has unseen speakers, but Kathbath's train split is in our training data and very likely the others'. DiarBench is real conversation from outside our training data. DiarBench has 200 turns from 44 conversations, 5,881 reference words, 659 of them English. The "English in English" column scores against references that write English words in English (Latin script), and English words count only if written in English letters; the native column scores against references that write those English words in Gurmukhi. Whisper's rates pass 100% because it adds words and often writes the wrong script, so they say little about Whisper at its best. Paired 95% intervals are Shruti Punjabi minus the other model, in word error points: negative means fewer errors for Shruti Punjabi, and an interval that includes 0 shows no lead either way.

- **Dolphin base:** FLEURS -57.30 to -50.97 by clip; Kathbath test-unknown -56.58 to -49.66 by clip; DiarBench native -63.82 to -58.09 by clip, -65.38 to -56.09 by conversation, -59.94 to -54.14 with script ignored; DiarBench English in English -68.65 to -62.72 by clip, -70.47 to -60.74 by conversation, -65.06 to -58.97 with script ignored.
- **IndicConformer Punjabi:** FLEURS +10.77 to +14.14 by clip; Kathbath test-unknown +3.33 to +6.42 by clip; DiarBench native +7.50 to +12.17 by clip, +5.53 to +13.10 by conversation, +9.81 to +13.89 with script ignored; DiarBench English in English -4.44 to +0.15 by clip, -6.37 to +0.68 by conversation, -2.69 to +1.53 with script ignored.
- **Dolphin small:** FLEURS -36.57 to -29.21 by clip; Kathbath test-unknown -36.49 to -28.76 by clip; DiarBench native -63.75 to -58.01 by clip, -65.47 to -55.92 by conversation, -64.75 to -59.14 with script ignored; DiarBench English in English -69.40 to -63.60 by clip, -71.21 to -61.45 by conversation, -70.53 to -64.75 with script ignored.
- **SraVaani-1.0:** FLEURS +12.35 to +16.34 by clip; Kathbath test-unknown +4.27 to +7.45 by clip; DiarBench native +12.20 to +16.45 by clip, +11.42 to +16.45 by conversation, +13.32 to +17.47 with script ignored; DiarBench English in English -1.34 to +2.78 by clip, -2.30 to +3.06 by conversation, -0.70 to +3.33 with script ignored.
- **English words in English letters:** 58.7% of DiarBench's 659 English words, on 200 turns. IndicConformer Punjabi and SraVaani-1.0 write none.
- **English:** LibriSpeech test-clean 6.24 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Punjabi speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 10.7x faster than real time, 64 MB peak memory (20 clips).

## Where it falls short

- **It leads no serious rival.** Its strongest result is a level one: the interval against IndicConformer Punjabi only just includes 0 (-4.44 to +0.15), and with script ignored it is fully level (-2.69 to +1.53).
- **Read Punjabi is far behind.** FLEURS 29.3, against 16.9 and 15.1; Kathbath test-unknown 18.6, against 13.7 and 12.7.
- **The native reference shows the gap.** DiarBench native 38.3, against 28.4 and 23.9. Character errors there 28.26%, against 13.84% and 13.42%.
- **Less English in this test.** 659 of 5,881 reference words, so English helps less here. Local names glossed in Latin often come out in Gurmukhi and count as errors (chandigarh as ਚੰਡੀਗੜ੍ਹ).
- **English words in Gurmukhi:** about 168 of DiarBench's 659 by automatic alignment (punjabi as ਪੰਜਾਬੀ). DiarBench also glosses some local names in Latin, so a correct native spelling can score as an error.
- **Browser:** tested on 9 lines only, played into the page on one phone's Chrome with the language set. All 9 came out in Gurmukhi, and 6 kept some English letters. Not tested with a live microphone. On the other Shruti languages where we measured it, plain English loses a little accuracy with the language set.
- **No Punjabi reader has reviewed this output yet.**
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Punjabi).
- **Not tested on Punjabi:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
