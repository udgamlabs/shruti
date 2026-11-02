# Shruti Tamil

Tamil and English speech to text, offline, from one 29.7 MB file (`shruti-tamil.cact`, 55.9M parameters). Tamil comes
out in Tamil script and English words stay in English. Built on Cactus Whistle via Shruti Hindi; Apache-2.0. Weights
and model card: [huggingface.co/udgamlabs/shruti-tamil](https://huggingface.co/udgamlabs/shruti-tamil).

## Results

Word error rate (%, lower is better), 200 clips per set, the same clips and scorer for every model; Shruti Tamil in its
on-device engine on a Mac. Measured 2 November 2026. Size is the file as published.

| Model | File size | FLEURS (read) | Kathbath test-unknown (read) | DiarBench (conversation, held out) |
|---|---|---|---|---|
| **Shruti Tamil** | **29.7 MB** | 38.5 | 33.2 | **52.5** |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 43.5 MB | 226.7 | 239.6 | 244.8 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 81.8 MB | 124.5 | 108.8 | 152.5 |
| [Dolphin base](https://huggingface.co/DataoceanAI/dolphin-base) (int8) | 103.7 MB | 71.9 | 65.0 | 92.4 |
| [Thedal v11](https://huggingface.co/Badri0510/Thedal-AED-Tamil-Tanglish-English-INT4-ConvINT8-ONNX) (non-commercial licence) | 102.6 MB | 42.2 | 38.1 | 70.1 |
| [IndicConformer Tamil](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-ta-onnx) (int8) | 137.7 MB | **27.8** | **25.0** | 58.7 |
| [deepdml whisper-tiny-ta](https://huggingface.co/deepdml/whisper-tiny-ta-mix-norm) (fp32) | 151.1 MB | 47.9 | 45.6 | 90.3 |
| [deepdml whisper-base-ta](https://huggingface.co/deepdml/whisper-base-ta-mix-norm) (fp32) | 290.4 MB | 38.7 | 39.1 | 81.1 |

DiarBench is real conversation that mixes in English; its references here write English words in English (Latin
script), as Shruti Tamil does. With the English words dropped from the reference, IndicConformer Tamil is ahead (51.4 vs
62.2), and so is the much larger SraVaani-1.0 (908.8 MB; 44.0). FLEURS and Kathbath are read Tamil.

- **Size class:** every other open model under 100 MB we measured gets more Tamil words wrong than right (Whisper adds
  words, so its rates pass 100%). On FLEURS, Shruti Tamil is level with deepdml's whisper-base-ta fine-tune (38.5 vs 38.7),
  which is 290 MB as published.
- **English inside Tamil:** Shruti Tamil writes 51.5% of DiarBench's English words in English; every other model above
  writes 8% or fewer.
- **English:** LibriSpeech test-clean 6.37 (Shruti Hindi: 5.95), Svarah (Indian English) 23.36.
- **Silence:** 0 words invented on 100 silent clips.
- **Held-out sets:** we did not train on FLEURS or DiarBench, or on Kathbath's unseen-speaker test set.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 10.8x faster than real time, 64 MB
peak memory (20 clips). In that phone's Chrome, a 3-7 s audio file is transcribed in about 1.5-3.8 s.

## Where it falls short

- **Pure Tamil accuracy.** IndicConformer Tamil (about 5x larger) makes about 28% fewer errors on FLEURS and 25% fewer on
  Kathbath; larger models such as SraVaani-1.0 fewer still, also on conversations scored on the native reference.
- **Numbers and loanwords:** nearby number words can be swapped (நூத்தம்பது, 150, came out as மூத்தம்பது) and some English
  loanwords come out in Tamil script ("meter" as மீட்டர்).
- **Spoken vs written forms** count as errors and are common (நேத்து/நேற்று, காலையில/காலையில்).
- **Language detection** can pick English for Tamil speech with many English words (one of our live phone takes came
  out garbled this way), and can label an all-English clip as Tamil (the text was still right).
- **Names it has never heard** are spelt by sound. Not tested: noise, music, far-away or speaker-played audio, Linux,
  Windows and iOS. No punctuation; numbers come out as words.

