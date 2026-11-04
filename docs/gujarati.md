# Shruti Gujarati

Gujarati and English speech to text, offline, from one 29.7 MB file (`shruti-gujarati.cact`). Gujarati comes out in
Gujarati script and English words stay in English. Built on Cactus Whistle via Shruti Hindi; Apache-2.0. Weights and model
card: [huggingface.co/udgamlabs/shruti-gujarati](https://huggingface.co/udgamlabs/shruti-gujarati).

## Results

Word error rate (%, lower is better), 200 clips per set, the same clips and scorer for every model; Shruti Gujarati in its
on-device engine on a Mac. Measured 2 November 2026. Size is the file as published (whisper.cpp q8_0 for Whisper).

| Model | File size | DiarBench (conversation, held out) | FLEURS (read) | Kathbath test-unknown (read) |
|---|---|---|---|---|
| **Shruti Gujarati** | **29.7 MB** | **28.5** | 27.9 | 20.7 |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 43.5 MB | 469.1 | 205.9 | 222.9 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 81.8 MB | 278.6 | 191.1 | 189.4 |
| [Dolphin base](https://huggingface.co/DataoceanAI/dolphin-base) (int8) | 103.7 MB | 77.6 | 74.7 | 64.0 |
| Vosk Gujarati (small, 0.42) | 108.1 MB | 71.9 | 36.5 | 28.0 |
| IndicConformer Gujarati (int8) | 137.7 MB | 44.9 | 18.8 | 14.5 |
| [Dolphin small](https://huggingface.co/DataoceanAI/dolphin-small) (int8) | 249.7 MB | 92.9 | 61.6 | 47.3 |
| [Whisper small](https://huggingface.co/openai/whisper-small) | 264.5 MB | 131.8 | 131.9 | 136.0 |
| [Whisper-base Gujarati](https://huggingface.co/vasista22/whisper-gujarati-base) (vasista22) | 290.5 MB | 72.5 | 32.3 | 29.8 |
| SraVaani-1.0 | 908.8 MB | 32.4 | **16.1** | **12.5** |
| [Whisper-small Gujarati](https://huggingface.co/vasista22/whisper-gujarati-small) (vasista22) | 967.1 MB | 65.4 | 26.8 | 35.1 |

DiarBench is real conversation that mixes in English; its references here write English words in English (Latin
script), as Shruti Gujarati does. With the English words dropped from the reference, SraVaani-1.0 is ahead (22.8; Shruti
Gujarati 33.6, IndicConformer Gujarati 40.2). FLEURS and Kathbath are read Gujarati.

- **Mixed Gujarati and English:** on DiarBench, Shruti Gujarati makes fewer errors than every model we measured, including
  SraVaani-1.0, about 30x its size (28.5 vs 32.4; paired 95% interval of the difference -5.66 to -2.14 points), and
  IndicConformer Gujarati, almost 5x its size (44.9; -19.03 to -13.91).
- **English inside Gujarati:** Shruti Gujarati writes 50.4% of DiarBench's English words in English; IndicConformer
  Gujarati, Vosk Gujarati, SraVaani-1.0 and the Whisper Gujarati fine-tunes write none.
- **Against Vosk Gujarati** (over three times its size): fewer errors on every set.
- **One file for both languages:** Gujarati and English in 29.7 MB.
- **English:** LibriSpeech test-clean 6.17 (Shruti Hindi: 5.95).
- **Silence:** 0 words invented on 100 silent clips.
- **Held-out sets:** we did not train on FLEURS or DiarBench, or on Kathbath's unseen-speaker test set.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 10.3x faster than real time, 64 MB
peak memory (20 clips).

## Where it falls short

- **Read Gujarati accuracy.** IndicConformer Gujarati and SraVaani-1.0 are clearly more accurate on read speech (FLEURS
  18.8 and 16.1 vs 27.9; Kathbath 14.5 and 12.5 vs 20.7), and the Whisper-small Gujarati fine-tune is level with it on
  FLEURS (26.8).
- **English words in Gujarati script:** some English words still come out in Gujarati script (party plot as પાર્ટી પ્લોટ,
  stock as સ્ટોક).
- **Real errors and spelling variants:** real errors happen (ઉછીની, "on loan", as ઓછીની), and joins and spelling variants
  count as errors (ચણિયાચોળી as ચણિયા ચોળી).
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family
  models, not on Gujarati).
- **Not tested on Gujarati:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation;
  numbers come out as words.
