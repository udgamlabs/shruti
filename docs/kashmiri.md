# Shruti Kashmiri

Kashmiri in Perso-Arabic script and English in one 29.7 MB file that runs on your device. File: `shruti-kashmiri.cact`. Kashmiri comes out in Perso-Arabic script, and English words can stay in English letters, as here: اَز چھُ واریاہ تُرُن heater چھُنہٕ چَلان. Built on Cactus Whistle via Shruti Hindi; Apache-2.0 for this model. Weights and model card: [huggingface.co/udgamlabs/shruti-kashmiri](https://huggingface.co/udgamlabs/shruti-kashmiri).

**Its one lead is on the set that shares a source with its training data.** On IndicVoices valid it has fewer word errors than IndicConformer Kashmiri (43.0 vs 48.7), but that set flatters both models, and IndicConformer Kashmiri has fewer character errors there (15.71% vs 23.16%). The independent check, DiarBench, is 32 turns with no English in its reference, too small to agree or disagree (56.4 vs 58.2, an interval that includes 0). No Kashmiri reader has reviewed this output yet.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Kashmiri in its on-device engine on a Mac. Measured 3 November 2026. Size is the file as published.

| Model | File size | IndicVoices valid | DiarBench |
|---|---|---|---|
| **Shruti Kashmiri** | 29.7 MB | 43.0 | 56.4 |
| [IndicConformer Kashmiri](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-ks-onnx) (int8) | 137.7 MB | 48.7 | 58.2 |

DiarBench Kashmiri's reference holds no English glosses, so its native and English-in-English references are the same; only one DiarBench column is shown. IndicVoices valid is IndicVoices' own held-out split: its speakers are kept out of training and its sentences blocked, but it is the same source as our training data, so it flatters us. DiarBench is real conversation from outside it: 32 turns from 14 conversations, 748 reference words, 0 of them English, in 0 turns. Paired 95% intervals are Shruti Kashmiri minus the other model, in word error points: negative means fewer errors for Shruti Kashmiri, and an interval that includes 0 shows no lead either way.

- **IndicConformer Kashmiri:** IndicVoices valid -8.39 to -3.17 by clip; DiarBench native -7.93 to +4.45 by clip, -9.08 to +3.85 by conversation.
- **English inside Kashmiri:** on IndicVoices valid it writes 75.6% of glossed English words in English letters (303 words). DiarBench Kashmiri has no English to test.
- **English:** LibriSpeech test-clean 6.08 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Kashmiri speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 8.6x faster than real time, 64 MB peak memory (20 clips).

## Where it falls short

- **Its one lead is on the flattering set.** IndicVoices valid shares a source with our training data. DiarBench agrees in direction only (56.4 vs 58.2; -7.93 to +4.45), on 32 turns.
- **Character errors are higher.** 23.16% vs 15.71% on IndicVoices valid, 34.9% vs 25.01% on DiarBench. Word error alone hides this.
- **High error rates.** 43.0 on IndicVoices valid and 56.4 on conversation: expect to correct a lot.
- **No English test on conversation.** DiarBench Kashmiri's reference holds no English words. On IndicVoices valid it writes 75.6% of glossed English words in English letters (303 words).
- **Perso-Arabic only.** It does not write Kashmiri in Devanagari.
- **Unmeasured rivals.** Kashmiri models have appeared within weeks, including `Aazu34/kashmiri-asr-ctc` (186.6 MB). We have not measured them.
- **Browser:** tested on 4 test clips only, played into the page on one phone's Chrome with the language set. All 4 came out in Perso-Arabic script. These are test clips read by an AI voice, not real speech. Not tested with a live microphone. On the other Shruti languages where we measured it, plain English loses a little accuracy with the language set.
- **No Kashmiri reader has reviewed this output yet.**
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Kashmiri).
- **Not tested on Kashmiri:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
