# Shruti Malayalam

Malayalam and English in one 29.7 MB file that runs on your device (`shruti-malayalam.cact`). Malayalam comes out in Malayalam script,
and about 4 in 10 English words stay in English letters. Built on Cactus Whistle via Shruti Hindi; Apache-2.0. Weights and model card:
[huggingface.co/udgamlabs/shruti-malayalam](https://huggingface.co/udgamlabs/shruti-malayalam).

**We make no headline comparison for this model.** IndicConformer Malayalam and SraVaani-1.0 are more accurate on read Malayalam and
against the native reference, and more than half of the words are wrong on conversations.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Malayalam in its on-device engine on a Mac.
FLEURS, Kathbath and DiarBench have 200 clips or turns each. Measured 2 November 2026. Size is the file as published (whisper.cpp q8_0
for Whisper).

| Model | File size | FLEURS (read) | Kathbath test-unknown (read) | DiarBench, English in English (conversation, held out) | DiarBench, native reference |
|---|---|---|---|---|---|
| **Shruti Malayalam** | 29.7 MB | 36.0 | 44.9 | 56.8 | 65.4 |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 43.5 MB | 509.4 | 910.2 | 555.0 | 559.6 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 81.8 MB | 186.3 | 275.1 | 217.8 | 219.4 |
| [IndicConformer Malayalam](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-ml-onnx) (int8) | 137.7 MB | 31.6 | 35.0 | 61.7 | 53.2 |
| [Whisper small](https://huggingface.co/openai/whisper-small) | 264.5 MB | 316.8 | 251.8 | 238.9 | 244.7 |
| [SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | 908.8 MB | 22.6 | 30.4 | 55.2 | 45.2 |

FLEURS is read speech we did not train on. Kathbath test-unknown has unseen speakers, but Kathbath's training split is in our training
data. DiarBench is real conversation: 200 turns from 34 conversations, 1,038 English words in 4,044. Its "English in English" references
write English words in English (Latin script), and English words count only if written in English letters; its native references write
those English words in Malayalam script. Whisper's rates pass 100% because it adds words and often writes the wrong script.

- **IndicConformer Malayalam:** ahead on FLEURS (paired 95% interval of the difference +1.07 to +7.39 points), Kathbath (+6.73 to +13.06)
  and DiarBench native (+9.49 to +15.18). On DiarBench English in English: -8.28 to -1.79 by clip, but -9.45 to +1.23 with whole
  conversations resampled and -4.48 to +1.50 with script ignored, so that lead is weak.
- **SraVaani-1.0:** ahead on FLEURS (+11.12 to +15.72), Kathbath (+11.36 to +17.53) and DiarBench native (+17.56 to +23.18). On DiarBench
  English in English, level as written (-1.29 to +4.81) and ahead with script ignored (+0.43 to +6.27).
- **English words in English letters:** 39.1% of DiarBench's 1,038. Whisper small 27.1%, Whisper tiny 15.9%, Whisper base 6.6%, SraVaani-1.0
  4.7%, IndicConformer Malayalam none.
- **English:** LibriSpeech test-clean 6.24 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Malayalam speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 10.1x faster than real time, 64 MB peak memory (20
clips).

## Where it falls short

- **Larger models are more accurate.** See above.
- **More than half of the words are wrong on conversations** (56.8% with English in English, 65.4% against the native reference). Malayalam
  fuses words, so word scores look harsher than the text reads (Kathbath: 44.9% of words, 9.0% of characters), but on DiarBench native
  character errors are 35.77%, against IndicConformer Malayalam's 19.03% and SraVaani-1.0's 16.03%. Expect to correct a lot.
- **English words often come out in Malayalam script:** about 336 of DiarBench's 1,038 (cinema as സിനിമ). This is the lowest share in English
  letters of the Shruti single-language models so far.
- **Browser:** tested on nine lines only, played into the page on one phone's Chrome with the language set. Not tested with a live
  microphone.
- **No Malayalam reader has reviewed this output yet.**
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on
  Malayalam).
- **Not tested on Malayalam:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
