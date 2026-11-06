# Shruti Assamese

Assamese and English in one 29.7 MB file (`shruti-assamese.cact`). On Assamese conversations that mix in English, fewer errors than
IndicConformer-600M and SraVaani-1.0, models about 86 and 31 times its size. Assamese comes out in Assamese script, with its own ৰ and ৱ,
and more than half of the English words stay in English letters. Built on Cactus Whistle via Shruti Hindi; Apache-2.0 for this model.
Weights and model card: [huggingface.co/udgamlabs/shruti-assamese](https://huggingface.co/udgamlabs/shruti-assamese).

**The result rests on English letters.** Against the native reference on the same conversations, and on read Assamese, both of those models
are well ahead. We did not measure a per-language IndicConformer Assamese.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Assamese in its on-device engine on a Mac. FLEURS
has 200 clips, DiarBench 84 turns. Measured 2 November 2026. Size is the file as published (whisper.cpp q8_0 for Whisper). Whisper small was
not run on Assamese.

| Model | File size | FLEURS (read) | DiarBench, English in English (conversation, held out) | DiarBench, native reference |
|---|---|---|---|---|
| **Shruti Assamese** | 29.7 MB | 35.1 | **30.2** | 43.2 |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 43.5 MB | 315.7 | 255.8 | 262.1 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 81.8 MB | 191.9 | 328.1 | 337.1 |
| [SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | 908.8 MB | 21.9 | 37.9 | 29.1 |
| [IndicConformer-600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | 2,560 MB | **20.2** | 39.8 | **25.9** |

FLEURS is read speech we did not train on. DiarBench is real conversation: 84 turns from 19 conversations, 375 English words in 1,545. Its
"English in English" references write English words in English (Latin script), and English words count only if written in English letters; its
native references write those English words in Assamese script. Whisper's rates pass 100% because it adds words and often writes the wrong
script.

- **IndicConformer-600M:** on DiarBench English in English, paired 95% interval of the difference -16.37 to -3.74 points by clip, -15.81 to -3.30
  with whole conversations resampled, -15.20 to -3.18 with script ignored. Ahead of us on FLEURS (+13.13 to +16.74) and DiarBench native (+12.51
  to +22.43).
- **SraVaani-1.0:** on DiarBench English in English -13.26 to -3.11 by clip, -12.20 to -3.59 by conversation, -12.53 to -2.76 with script ignored.
  Ahead of us on FLEURS (+11.38 to +15.06) and DiarBench native (+10.45 to +17.89).
- **English words in English letters:** 57.3% of DiarBench's 375. IndicConformer-600M none, SraVaani-1.0 8.5%, Whisper base 48.5%, Whisper tiny 34.9%.
- **English:** LibriSpeech test-clean 6.22 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Assamese speech models exist that we have not measured, including per-language IndicConformer Assamese copies; these results cover only
the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 10.7x faster than real time, 64 MB peak memory (20 clips).

## Where it falls short

- **Read Assamese is far behind:** FLEURS 35.1, against 20.2 and 21.9. It is the only read test here.
- **The lead rests on English letters.** DiarBench native 43.2, against 25.9 and 29.1. Character errors there 26.69%, against 8.18% and 9.27%.
- **A small conversation test.** No Assamese reader has reviewed this output yet.
- **Two training sets name no licence.** SPRING-INX Assamese R1 and R2 state none on their pages (checked 6 November 2026); the
  [SPRING-INX paper](https://arxiv.org/abs/2310.14654) says "we are open sourcing SPRING-INX data". We kept them in training and don't claim a licence
  for them. See the model card.
- **English words in Assamese script:** about 69 of DiarBench's 375 (option as অপশ্যন). 3 of 84 turns came out wholly in English letters.
- **Browser:** tested on seven lines only, played into the page on one phone's Chrome with the language set. Not tested with a live microphone.
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Assamese).
- **Not tested on Assamese:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
