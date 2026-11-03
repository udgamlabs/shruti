# Shruti Telugu

Telugu and English speech to text, offline, from one 29.7 MB file (`shruti-telugu.cact`). Telugu comes out in Telugu
script and English words stay in English. Built on Cactus Whistle via Shruti Hindi; Apache-2.0. Weights and model card:
[huggingface.co/udgamlabs/shruti-telugu](https://huggingface.co/udgamlabs/shruti-telugu).

## Results

Word error rate (%, lower is better), 200 clips per set, the same clips and scorer for every model; Shruti Telugu in its
on-device engine on a Mac. Measured 2 November 2026. Size is the file as published (whisper.cpp q8_0 for Whisper).

| Model | File size | DiarBench (conversation, held out) | FLEURS (read) | Kathbath test-unknown (read) |
|---|---|---|---|---|
| **Shruti Telugu** | **29.7 MB** | **45.4** | 38.0 | 32.9 |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 43.5 MB | 659.1 | 687.8 | 583.3 |
| Vosk Telugu (small, 0.42) | 60.5 MB | 93.8 | 77.1 | 80.1 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 81.8 MB | 371.3 | 220.1 | 363.3 |
| [Dolphin base](https://huggingface.co/DataoceanAI/dolphin-base) (int8) | 103.7 MB | 86.9 | 73.4 | 66.2 |
| IndicConformer Telugu (int8) | 137.7 MB | 52.9 | 24.5 | 23.0 |
| [Whisper-tiny Telugu](https://huggingface.co/vasista22/whisper-telugu-tiny) (vasista22) | 151.1 MB | 59.2 | 34.2 | 34.4 |
| [Whisper-base Telugu](https://huggingface.co/vasista22/whisper-telugu-base) (vasista22) | 290.5 MB | 55.3 | 28.9 | 29.5 |
| SraVaani-1.0 | 908.8 MB | 47.8 | **22.0** | **20.3** |
| [Whisper-small Telugu](https://huggingface.co/vasista22/whisper-telugu-small) (vasista22) | 967.1 MB | 55.7 | 24.0 | 26.0 |

DiarBench is real conversation that mixes in English; its references here write English words in English (Latin
script), as Shruti Telugu does. With the English words dropped from the reference, most larger models are ahead
(SraVaani-1.0 33.0, IndicConformer Telugu 37.3, Shruti Telugu 58.0). FLEURS and Kathbath are read Telugu.

- **Mixed Telugu and English:** on DiarBench, Shruti Telugu makes fewer errors than IndicConformer Telugu (almost 5x its
  size), 45.4 vs 52.9, and than the Whisper Telugu fine-tunes (up to 32x its size), 55.7 for the largest. SraVaani-1.0,
  about 30x its size, is level with it (47.8; the paired interval includes zero).
- **English inside Telugu:** Shruti Telugu writes 53.7% of DiarBench's English words in English; IndicConformer Telugu,
  Vosk Telugu and the Whisper Telugu fine-tunes write none.
- **Against Vosk Telugu** (about twice its size): about half the errors on every set.
- **One file for both languages:** Telugu and English in 29.7 MB; the Vosk and IndicConformer models are Telugu only.
- **English:** LibriSpeech test-clean 6.22 (Shruti Hindi: 5.95).
- **Silence:** 0 words invented on 100 silent clips.
- **Held-out sets:** we did not train on FLEURS or DiarBench, or on Kathbath's unseen-speaker test set.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 9.8x faster than real time, 64 MB
peak memory (20 clips).

## Where it falls short

- **Read Telugu accuracy.** IndicConformer Telugu, SraVaani-1.0 and the Whisper-base and Whisper-small Telugu fine-tunes
  are clearly more accurate on read speech (FLEURS 24.5, 22.0, 28.9 and 24.0 vs 38.0), and the Whisper-tiny Telugu
  fine-tune is ahead on FLEURS (34.2).
- **English words in Telugu script:** some English words still come out in Telugu script (show as షోకి, online as
  ఆన్‌లైన్‌లో), and joins and spelling variants count as errors (తంగేడు పూలు as తంగేడుపూలు, ఆడుదాం as ఆడదాం).
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family
  models, not on Telugu).
- **Names it has never heard** are spelt by sound. Not tested on Telugu: noise, music, far-away or speaker-played audio,
  Linux, Windows and iOS. No punctuation; numbers come out as words.
