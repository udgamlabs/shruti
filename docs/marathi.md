# Shruti Marathi

Marathi and English speech to text, offline, from one 29.7 MB file (`shruti-marathi.cact`, 55.7M parameters). Marathi
comes out in Devanagari and English words stay in English: *उद्या सकाळी नऊ वाजता physics ची exam आहे*. Built on Cactus
Whistle via Shruti Hindi; Apache-2.0. Weights and model card:
[huggingface.co/udgamlabs/shruti-marathi](https://huggingface.co/udgamlabs/shruti-marathi).

## Results

Word error rate (%, lower is better), 200 clips per set. Every model was run by us on the same clips with the same text
normaliser and scorer; Shruti Marathi in its on-device engine on a Mac, the others as listed in
"How each model was run" below. Measured 30 October to 2 November 2026. Size is the file as published.

| Model | File size | FLEURS (read) | Kathbath (read) | DiarBench (conversation, held out) | IndicVoices | SPRING-INX (phone calls) | Vaani (spontaneous) |
|---|---|---|---|---|---|---|---|
| **Shruti Marathi** | **29.7 MB** | 29.9 | 24.5 | **38.2** | **22.2** | **42.9** | **32.2** |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 44 MB | 273.1 | 278.1 | 373.9 | 321.8 | 375.6 | 338.7 |
| [Whisper base](https://huggingface.co/openai/whisper-base) | 82 MB | 177.9 | 193.2 | 402.9 | 450.2 | 420.6 | 412.3 |
| [Dolphin base](https://huggingface.co/DataoceanAI/dolphin-base) (int8) | 103.7 MB | 71.6 | 74.0 | 97.0 | 95.8 | 96.2 | 94.4 |
| [marathi-conformer-33m-moe](https://huggingface.co/rimrafadii/marathi-conformer-33m-moe) (fp32) | 136.7 MB | 41.0 | 32.2 | 76.7 | 66.0 | 77.3 | 64.0 |
| [IndicConformer Marathi](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-mr-onnx) (int8) | 137.7 MB | **22.4** | **14.1** | 41.5 | 27.6 | 53.0 | 44.0 |

DiarBench, IndicVoices, SPRING-INX and Vaani are conversations that mix in English. Their references here write English
words in English (Latin script), as Shruti Marathi does; models that write English in Devanagari are penalised for it.
With English words written in Devanagari in the references (the "native" view), IndicConformer Marathi is ahead: DiarBench
32.3 vs 40.9, IndicVoices 21.3 vs 26.6, Vaani 37.6 vs 41.7. FLEURS and Kathbath have one Devanagari reference.

- **Size class:** every other open model under 100 MB we measured gets more Marathi words wrong than right (Whisper tiny
  and base add words too, so their rates pass 100%). Shruti Marathi is within 1.33x of IndicConformer Marathi's error
  rate on FLEURS (29.9 vs 22.4) and 1.73x on Kathbath (24.5 vs 14.1), at about a fifth of its size.
- **English inside Marathi:** Shruti Marathi writes 40-76% of the English words in English on the four mixed sets; the
  other models in the table write none (Whisper tiny and base write some, among many errors).
- **English:** LibriSpeech test-clean 6.23 (Shruti Hindi: 5.95), Svarah (Indian English) 23.46.
- **Silence:** 0 words invented on 100 silent clips (marathi-conformer-33m-moe: 75).
- **Held-out sets:** we did not train on FLEURS or DiarBench. IndicVoices, Kathbath, SPRING-INX and Vaani share corpora
  with our training data; their test speakers and every test sentence were kept out. IndicConformer also trained on
  Kathbath and IndicVoices. Kathbath's public valid split stands in for its test split.

**Speed and memory** (offline):

| Device | Result |
|---|---|
| Samsung Galaxy F62 (2021, Exynos 9825, 6 GB RAM, Android 13), native engine | 12.1x faster than real time, 64 MB peak memory (20 clips); text identical to the Mac on 16 of 20 |
| Galaxy F62, Chrome, the browser demo (WebAssembly, one thread) | a 3-5 s audio file transcribed in 1.4-2.8 s |
| Desktop Chrome on a Mac (WebAssembly) | a 5.0 s clip in 0.4 s |

## Where it falls short

- **Pure Marathi accuracy.** IndicConformer Marathi (about 5x larger) makes about a quarter fewer errors on FLEURS and
  about 40% fewer on Kathbath; 0.6B models such as Hima-ASR and IndicConformer-600M (40-90x larger) fewer still. Choose
  them if you only need Marathi in Devanagari and have the memory.
- **Spelling variants** count as errors and are common: उशिरा/उशीरा, words joined or split (वेळेवर घे → वेळेवर्गे).
- **English loanwords** sometimes come out in Devanagari (फेसिक्स for "physics" with one of our two test voices), and
  acronyms split into letters (UPI → "u p आय").
- **Not Hindi.** It was retrained for Marathi; Hindi speech comes out as Marathi-like text. Use Shruti Hindi for Hindi.
- **English** is a little weaker than Shruti Hindi's (LibriSpeech 6.23 vs 5.95), and far behind large English models.
- **Names it has never heard** are spelt by sound. Pass names and terms as `keywords` to help.
- **Not tested:** noisy audio, music, far-away or speaker-played audio, and Linux, Windows and iOS (the engine runs
  there). Long audio is processed in 30 s windows. No punctuation; numbers come out as words.


## How each model was run

Same 1,200 clips for every model: 200 per set, the first clips of 1-29 s in a fixed seeded order (SPRING-INX: a fixed
set of 200 code-mixed clips). Same scorer: Unicode NFC, lowercase, punctuation to spaces, word error rate with a
1,000-resample bootstrap 95% interval, plus a script-blind romanised view and English recall.

| Set | What it is | Licence | References |
|---|---|---|---|
| FLEURS mr_in test (Google) | read Wikipedia sentences | CC-BY-4.0 | Devanagari |
| Kathbath Marathi valid (AI4Bharat) | read sentences over the phone; the public valid split stands in for the test split | CC-BY-4.0 | Devanagari |
| DiarBench Marathi (Sarvam, AI4Bharat) | real conversations, single-speaker turns with at least one English word | CC-BY-4.0 | English in Latin; and native |
| IndicVoices Marathi valid (AI4Bharat) | read, extempore and conversational | CC-BY-4.0 | English in Latin (from the dataset's own English tags); and native |
| SPRING-INX R2 Marathi test (IIT Madras) | phone conversations | CC-BY-4.0 | as written (English in Latin) |
| Vaani Marathi test (ARTPARK-IISc) | image-prompted spontaneous speech | CC-BY-4.0 | English in Latin; and native |

| Model | How it ran |
|---|---|
| Shruti Marathi | the shipped `shruti-marathi.cact` in the Cactus needle engine on a Mac (Apple silicon), 5-beam, language detected |
| Whisper tiny, base | transformers fp16 on a GPU, beam 5, Marathi set (size: whisper.cpp q8_0 files) |
| Dolphin base | sherpa-onnx int8 CTC on a Mac; the export picks its own language |
| marathi-conformer-33m-moe | the published fp32 checkpoint on a Mac CPU, greedy CTC; the published code lacks its MoE layer class, so we rebuilt it from the checkpoint |
| IndicConformer Marathi | the OpenVoiceOS int8 ONNX export (AI4Bharat's 120M per-language model), onnx-asr greedy CTC on a CPU |

Whisper runs away under plain beam search without temperature fallback (output longer than twice the reference on many
clips), which inflates its numbers; Whisper's own Marathi fine-tune (whisper-small-marathi, 967 MB) still scores 55.1 on
FLEURS and 54.0 on Kathbath in the same harness.
