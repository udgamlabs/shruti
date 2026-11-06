# Shruti Urdu

Urdu and English speech to text, offline, from one 29.7 MB file (`shruti-urdu.cact`). Urdu comes out in Perso-Arabic script
and English words stay in English. Built on Cactus Whistle via Shruti Hindi; Apache-2.0. Weights and model card:
[huggingface.co/udgamlabs/shruti-urdu](https://huggingface.co/udgamlabs/shruti-urdu).

<div dir="rtl" lang="ur">میری bike کا tyre puncture ہو گیا mechanic کو phone کیا ہے آدھے گھنٹے میں آ جائے گا</div>

Show the transcript right to left (`dir="rtl"`); the English words keep their own direction inside the line.

**The result below rests on English letters.** Against the native reference and on read Urdu, IndicConformer Urdu is ahead.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Urdu in its on-device engine on a
Mac. Measured 2 November 2026. FLEURS and Kathbath have 200 clips each, DiarBench 74 turns. Size is the file as published
(whisper.cpp q8_0 for Whisper).

| Model | File size | DiarBench (conversation, held out) | FLEURS (read) | Kathbath test-unknown (read) |
|---|---|---|---|---|
| **Shruti Urdu** | **29.7 MB** | **22.9** | 33.5 | 18.2 |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 43.5 MB | 257.3 | 108.8 | 107.4 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 81.8 MB | 111.7 | 55.6 | 55.3 |
| [Dolphin base](https://huggingface.co/DataoceanAI/dolphin-base) (int8) | 103.7 MB | 98.5 | 63.0 | 57.5 |
| [IndicConformer Urdu](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-ur-onnx) (int8) | 137.7 MB | 29.1 | **24.7** | **13.4** |
| [Dolphin small](https://huggingface.co/DataoceanAI/dolphin-small) (int8) | 249.7 MB | 98.5 | 75.3 | 51.3 |
| [Whisper small](https://huggingface.co/openai/whisper-small) | 264.5 MB | 50.8 | 41.7 | 45.7 |

DiarBench is real conversation that mixes in English: 74 turns from 25 conversations, 251 English words in 1,704. Its
references here write English words in English (Latin script), as Shruti Urdu does, so English words count only if they are
written in English letters. With references that write those words in Urdu script instead, IndicConformer Urdu (19.5) is
ahead of Shruti Urdu (31.3). FLEURS and Kathbath are read Urdu.

- **Mixed Urdu and English:** on DiarBench, fewer errors than IndicConformer Urdu (22.9 vs 29.1; paired 95% interval of the
  difference -10.22 to -1.64 points; with whole conversations resampled -9.99 to -1.51) and than Whisper small (50.8).
- **English inside Urdu:** Shruti Urdu writes 64.1% of DiarBench's English words in English; IndicConformer Urdu writes none,
  Whisper small 10.8%, Whisper base 6.4%, Dolphin base and small 3.2%.
- **One file for both languages:** Urdu and English in 29.7 MB.
- **English:** LibriSpeech test-clean 6.07 (Shruti Hindi: 5.95).
- **Silence:** 0 words invented on 100 silent clips.
- **Held-out sets:** we did not train on FLEURS or DiarBench, or on Kathbath's unseen-speaker test set.

Other Urdu speech models exist that we have not measured, and these results cover only the models listed here. A scan of
Hugging Face on 6 November 2026 found, among others,
[Urdu-Streaming-Zipformer-20M](https://huggingface.co/mahwizzzz/Urdu-Streaming-Zipformer-20M) (43.9 MB as int8 encoder plus
decoder; gated, so we have not run it).

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, about 11x faster than real time, 64 MB
peak memory (20 clips).

## Where it falls short

- **The lead rests on English letters.** Against the native reference IndicConformer Urdu is well ahead on the same turns
  (19.5 vs 31.3; +8.26 to +16.02). We cannot show that Shruti Urdu does better beyond writing English in English letters.
- **Read Urdu accuracy.** IndicConformer Urdu makes about 26% fewer errors on read speech (FLEURS 24.7 vs 33.5; Kathbath 13.4
  vs 18.2).
- **Character errors.** On DiarBench's native reference, 28.69% against IndicConformer Urdu's 7.40%, almost four times.
- **A small test.** 74 turns is a small set; the interval of -10.22 to -1.64 does not pin down the size of the lead.
- **Who it was measured on.** No test set here is checked to be speakers from Pakistan (FLEURS carries an `ur_pk` label, and
  that is all we know), and the training collections' cards don't document speakers from Pakistan. No Urdu reader has reviewed
  this output yet.
- **English words in Urdu script:** about a third of DiarBench's English words don't come out in English letters; about 41 of 251
  are written in Urdu script (delhi as دہلی).
- **Real errors and spelling variants:** real errors happen (ساتھ رکھ as سات ہٹ), and joins and spelling variants count as errors.
- **In a browser, set the language.** In the phone's browser, with the language left to the engine, 1 of 8 lines we played in on
  5 November came out as English words for the whole line; the native engine detected all of them correctly. Passing `ur` is what
  our try box and the browser example do: 3 of 3 lines played in that way on 6 November stayed in Urdu script with English in
  Latin, and that line stayed in Urdu script in a Mac's browser, but has not been played again on the phone. On the other Shruti languages where we measured it, plain English loses
  a little accuracy this way. Not yet tested with a live microphone.
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models,
  not on Urdu).
- **Not tested on Urdu:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out
  as words.
