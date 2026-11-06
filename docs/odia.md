# Shruti Odia

Odia and English speech to text, offline, from one 29.7 MB file (`shruti-odia.cact`). Odia comes out in Odia script and
English words stay in English. Built on Cactus Whistle via Shruti Hindi; Apache-2.0. Weights and model card:
[huggingface.co/udgamlabs/shruti-odia](https://huggingface.co/udgamlabs/shruti-odia).

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Odia in its on-device engine on a
Mac. FLEURS and Kathbath have 200 clips each, DiarBench 196 turns. Measured 2 November 2026. Size is the file as published.

| Model | File size | DiarBench (conversation, held out) | FLEURS (read) | Kathbath test-unknown (read) |
|---|---|---|---|---|
| **Shruti Odia** | **29.7 MB** | **36.2** | 39.5 | 32.5 |
| [Dolphin base](https://huggingface.co/DataoceanAI/dolphin-base) (int8) | 103.7 MB | 97.5 | 68.1 | 52.7 |
| IndicConformer Odia (int8) | 137.7 MB | 49.9 | 25.9 | **21.1** |
| [Dolphin small](https://huggingface.co/DataoceanAI/dolphin-small) (int8) | 249.7 MB | 99.9 | 71.1 | 40.4 |
| SraVaani-1.0 | 908.8 MB | 43.4 | **22.1** | 21.4 |

DiarBench is real conversation that mixes in English; its references here write English words in English (Latin script),
as Shruti Odia does. With English words written in Odia script in the reference, SraVaani-1.0 (34.9) and IndicConformer Odia (40.7)
are ahead (Shruti Odia 49.2). FLEURS and Kathbath are read Odia. Whisper's language list has no Odia.

- **Mixed Odia and English:** on DiarBench, Shruti Odia makes fewer errors than every model we measured, including
  SraVaani-1.0, about 31x its size (36.2 vs 43.4; paired 95% interval of the difference -9.75 to -4.76 points), and
  IndicConformer Odia, almost 5x its size (49.9; -16.36 to -11.38).
- **English inside Odia:** Shruti Odia writes 55.8% of DiarBench's English words in English; IndicConformer Odia and
  SraVaani-1.0 write none.
- **Against Dolphin:** fewer errors on every set.
- **One file for both languages:** Odia and English in 29.7 MB.
- **English:** LibriSpeech test-clean 6.25 (Shruti Hindi: 5.95).
- **Silence:** 0 words invented on 100 silent clips.
- **Held-out sets:** we did not train on FLEURS or DiarBench, or on Kathbath's unseen-speaker test set.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 10.3x faster than real time, 64 MB
peak memory (20 clips).

## Where it falls short

- **Read Odia accuracy.** IndicConformer Odia and SraVaani-1.0 are clearly more accurate on read speech (FLEURS 25.9 and
  22.1 vs 39.5; Kathbath 21.1 and 21.4 vs 32.5). Character errors on FLEURS are 18.38% against 7.64% and 6.79%.
- **In a browser, set the language.** In the phone's browser, with the language left to the engine, 3 of 4 English-heavy lines we
  played in on 5 November came out as English garble. Passing `or` fixed them on 6 November, and it is what our try box does. Plain
  English loses a little accuracy this way. The native engine detected all of them correctly. Not yet tested with a live microphone.
- **English words in Odia script:** many English words come out in English letters, not all (start as ଷ୍ଟାର୍ଟ, online as
  ଅନଲାଇନ୍‌ରେ), and acronyms can split into letters (EMI as e m i).
- **Real errors and spelling variants:** real errors happen (Cyclone as ସାଇଟ୍ରୋନର, ମହମବତୀ, candles, as ମହମ୍ମଦୀ). Local names
  that DiarBench glosses in Latin come out in Odia (cuttack as କଟକ) and score as errors.
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family
  models, not on Odia).
- **Not tested on Odia:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers
  come out as words. No Odia speaker has reviewed this output yet.
