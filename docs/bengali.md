# Shruti Bengali

Bengali and English speech to text, offline, from one 29.7 MB file (`shruti-bengali.cact`). Bengali comes out in Bengali
script and English words stay in English. Built on Cactus Whistle via Shruti Hindi; Apache-2.0. Weights and model card:
[huggingface.co/udgamlabs/shruti-bengali](https://huggingface.co/udgamlabs/shruti-bengali).

## Results

Word error rate (%, lower is better), 200 clips per set, the same clips and scorer for every model; Shruti Bengali in its
on-device engine on a Mac. Measured 2 November 2026. Size is the file as published.

| Model | File size | DiarBench (conversation, held out) | FLEURS (read) | Kathbath test-unknown (read) |
|---|---|---|---|---|
| **Shruti Bengali** | **29.7 MB** | **36.0** | 29.9 | 22.2 |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 43.5 MB | 540.9 | 361.6 | 403.7 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 81.8 MB | 339.9 | 176.7 | 214.5 |
| Vosk Bengali (Zipformer) | 90.0 MB | 42.8 | 20.8 | 21.7 |
| [Dolphin base](https://huggingface.co/DataoceanAI/dolphin-base) (int8) | 103.7 MB | 93.0 | 41.1 | 45.9 |
| IndicConformer Bengali (int8) | 137.7 MB | 39.3 | 20.7 | **12.1** |
| SraVaani-1.0 | 908.8 MB | 34.6 | **15.7** | 11.3 |

DiarBench is real conversation that mixes in English; its references here write English words in English (Latin
script), as Shruti Bengali does. With the English words dropped from the reference, Vosk Bengali (38.5) and IndicConformer
Bengali (34.8) are ahead of Shruti Bengali (41.8). FLEURS and Kathbath are read Bengali.

- **Mixed Bengali and English:** on DiarBench, Shruti Bengali makes fewer errors than Vosk Bengali (about 3x its size) and
  IndicConformer Bengali (about 5x), 36.0 vs 42.8 and 39.3. SraVaani-1.0, about 30x its size, is ahead (34.6).
- **English inside Bengali:** Shruti Bengali writes 49.7% of DiarBench's English words in English; Vosk Bengali and
  IndicConformer Bengali write none.
- **One file for both languages:** Bengali and English in 29.7 MB, about a third of Vosk Bengali's size.
- **English:** LibriSpeech test-clean 6.30 (Shruti Hindi: 5.95).
- **Silence:** 0 words invented on 100 silent clips.
- **Held-out sets:** we did not train on FLEURS or DiarBench, or on Kathbath's unseen-speaker test set.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 12.6x faster than real time, 64 MB
peak memory (20 clips).

## Where it falls short

- **Read Bengali accuracy.** Vosk Bengali (90 MB) and IndicConformer Bengali are clearly more accurate on read speech
  (FLEURS 20.8 and 20.7 vs 29.9), and SraVaani-1.0 more so.
- **English words in Bengali script:** some English words still come out in Bengali script (derby as ডার্ভিটা, goal as
  গোলটা), and spelling variants count as errors (বেরোব/বেরোবো).
- **Names it has never heard** are spelt by sound. Not tested: noise, music, far-away or speaker-played audio, Linux,
  Windows and iOS. No punctuation; numbers come out as words.
