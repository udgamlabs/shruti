# Shruti Kannada

Kannada and English speech to text, offline, from one 29.7 MB file (`shruti-kannada.cact`). Kannada comes out in Kannada
script and English words stay in English. Built on Cactus Whistle via Shruti Hindi; Apache-2.0. Weights and model card:
[huggingface.co/udgamlabs/shruti-kannada](https://huggingface.co/udgamlabs/shruti-kannada).

**Over half of the words are wrong on the conversation test below:** 56.6% with English words scored in English letters,
69.5% against the native reference.

## Results

Word error rate (%, lower is better), 200 clips per set, the same clips and scorer for every model; Shruti Kannada in its
on-device engine on a Mac. Measured 2 November 2026. Size is the file as published (whisper.cpp q8_0 for Whisper).

| Model | File size | DiarBench (conversation, held out) | FLEURS (read) | Kathbath test-unknown (read) |
|---|---|---|---|---|
| **Shruti Kannada** | **29.7 MB** | **56.6** | 33.1 | 29.4 |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 43.5 MB | 574.5 | 305.5 | 371.7 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 81.8 MB | 248.8 | 179.5 | 178.5 |
| IndicConformer Kannada (int8) | 137.7 MB | 67.1 | 20.4 | 19.8 |
| [Whisper-tiny Kannada](https://huggingface.co/vasista22/whisper-kannada-tiny) (vasista22) | 151.1 MB | 91.4 | 29.7 | 34.2 |
| [Whisper small](https://huggingface.co/openai/whisper-small) | 264.5 MB | 118.9 | 103.0 | 100.8 |
| [Whisper-base Kannada](https://huggingface.co/vasista22/whisper-kannada-base) (vasista22) | 290.5 MB | 90.9 | 25.6 | 31.0 |
| SraVaani-1.0 | 908.8 MB | 60.7 | **19.1** | **15.3** |
| [Whisper-small Kannada](https://huggingface.co/vasista22/whisper-kannada-small) (vasista22) | 967.1 MB | 88.7 | 20.8 | 26.5 |

DiarBench is real conversation that mixes in English; its references here write English words in English (Latin script),
as Shruti Kannada does. With English words written in Kannada script in the reference, SraVaani-1.0 (46.4) and IndicConformer Kannada
(56.7) are ahead (Shruti Kannada 69.5). FLEURS and Kathbath are read Kannada.

- **Mixed Kannada and English:** on DiarBench, Shruti Kannada makes fewer errors than IndicConformer Kannada, almost 5x its
  size (56.6 vs 67.1; paired 95% interval of the difference -13.21 to -7.82 points), and the three vasista22 Whisper Kannada
  fine-tunes, up to 33x its size (88.7 to 91.4).
- **SraVaani-1.0 on that test:** as written, 56.6 vs 60.7, but the gap does not hold with whole conversations resampled
  (-11.19 to +1.64), and with script ignored the two are level (-2.87 to +2.34). We do not claim a lead over it.
- **English inside Kannada:** Shruti Kannada writes 53.1% of DiarBench's English words in English; IndicConformer Kannada,
  SraVaani-1.0 and the vasista22 fine-tunes write none.
- **One file for both languages:** Kannada and English in 29.7 MB.
- **English:** LibriSpeech test-clean 6.16 (Shruti Hindi: 5.95).
- **Silence:** 0 words invented on 100 silent clips.
- **Held-out sets:** we did not train on FLEURS or DiarBench, or on Kathbath's unseen-speaker test set.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 11.2x faster than real time, 64 MB
peak memory (20 clips).

## Where it falls short

- **Most of a real conversation can be wrong.** On DiarBench 56.6% of words are wrong with English in English letters, and
  69.5% against the native reference.
- **Read Kannada accuracy.** IndicConformer Kannada and SraVaani-1.0 are clearly more accurate on read speech (FLEURS 20.4
  and 19.1 vs 33.1; Kathbath 19.8 and 15.3 vs 29.4), and all three vasista22 fine-tunes are ahead on FLEURS.
- **In a browser, set the language.** In the phone's browser, with the language left to the engine, 1 of 4 English-heavy lines we
  played in came out as English words for the whole line. Passing `kn` fixed it on 6 November, and it is what our try box does. Plain
  English loses a little accuracy this way. The native engine detected all of them correctly. Not yet tested with a live microphone.
- **English words in Kannada script:** only about half of the English words come out in English letters; others are written
  in Kannada script (monday as ಮಂಡೆ, laptop as ಲ್ಯಾಪ್‌ಟಾಪ್‌ನ).
- **Real errors and spelling variants:** real errors happen (parcel as ಭಾರಿಸಲ್ಲ, hard disk as hardness), and joins and
  spelling variants count as errors.
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family
  models, not on Kannada).
- **Not tested on Kannada:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers
  come out as words. No Kannada speaker has reviewed this output yet.
