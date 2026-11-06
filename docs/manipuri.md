# Shruti Manipuri

Manipuri in Meetei Mayek and English in one 29.7 MB file that runs on your device. File: `shruti-manipuri.cact`. Manipuri comes out in Meetei Mayek, and about 60% of DiarBench's 243 English words stay in English letters: ꯍꯦꯡ ꯑꯌꯨꯛ ꯈꯨꯃꯟ ꯃꯄꯥꯛꯇ boxing practice ꯂꯩꯒꯅꯤ. Built on Cactus Whistle via Shruti Hindi; Apache-2.0 for this model. Weights and model card: [huggingface.co/udgamlabs/shruti-manipuri](https://huggingface.co/udgamlabs/shruti-manipuri).

**A pointer, not a result.** On 44 conversation turns from 20 conversations (DiarBench, held out, 39 turns with English words), counting English words only in English letters, it has fewer errors than IndicConformer Manipuri (48.2 vs 65.4), but that is under the size we'd trust, and the interval is wide. Against the native reference on the same turns IndicConformer Manipuri is far ahead (53.5 vs 66.0), and on IndicVoices valid too (31.2 vs 36.0). It writes Meetei Mayek, not Bengali script. No Manipuri reader has reviewed this output yet.

## Results

Word error rate (%, lower is better), the same clips and scorer for every model; Shruti Manipuri in its on-device engine on a Mac. Measured 3 November 2026. Size is the file as published.

| Model | File size | IndicVoices valid | DiarBench, English in English | DiarBench, native |
|---|---|---|---|---|
| **Shruti Manipuri** | 29.7 MB | 36.0 | 48.2 | 66.0 |
| [IndicConformer Manipuri](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-mni-onnx) (int8) | 137.7 MB | 31.2 | 65.4 | 53.5 |

IndicVoices valid is IndicVoices' own held-out split: its speakers are kept out of training and its sentences blocked, but it is the same source as our training data, so it flatters us. DiarBench is real conversation from outside it: 44 turns from 20 conversations, 815 reference words, 243 of them English, in 39 turns. The "English in English" column scores against references that write English words in English (Latin script), and English words count only if written in English letters; the native column scores against references that write those English words in Meetei Mayek. Paired 95% intervals are Shruti Manipuri minus the other model, in word error points: negative means fewer errors for Shruti Manipuri, and an interval that includes 0 shows no lead either way.

- **IndicConformer Manipuri:** IndicVoices valid +2.07 to +7.70 by clip; DiarBench native +7.02 to +18.67 by clip, +6.39 to +20.31 by conversation; DiarBench English in English -26.78 to -7.98 by clip, -31.04 to -8.05 by conversation.
- **English words in English letters:** 60.9% of DiarBench's 243 English words, on 44 turns. IndicConformer Manipuri writes none.
- **English:** LibriSpeech test-clean 5.97 (Shruti Hindi: 5.95). **Silence:** 0 words invented on 100 silent clips.

Other Manipuri speech models exist that we have not measured; these results cover only the models listed here.

**Speed and memory** (offline): Samsung Galaxy F62 (2021, 6 GB RAM), native engine, 11.2x faster than real time, 64 MB peak memory (20 clips).

## Where it falls short

- **A small test.** 44 turns from 20 conversations; the interval is wide (-26.78 to -7.98), and no script-ignored check exists for Meetei Mayek.
- **The native reference shows the gap.** DiarBench native 66.0 vs 53.5. Character errors there 40.29% vs 17.48%, the highest in this group. IndicVoices valid 36.0 vs 31.2.
- **Meetei Mayek only.** It does not write Manipuri in Bengali script.
- **Sometimes the whole line comes out in English letters:** 4 of 200 IndicVoices clips and 2 of 44 DiarBench turns, with the engine's own language choice.
- **English words in Meetei Mayek:** about 33 of DiarBench's 243 by automatic alignment. DiarBench also glosses some local names in Latin, so a correct native spelling can score as an error.
- **Browser:** tested on 4 test clips only, played into the page on one phone's Chrome with the language set. All 4 came out in Meetei Mayek. These are test clips read by an AI voice, not real speech. Not tested with a live microphone. On the other Shruti languages where we measured it, plain English loses a little accuracy with the language set.
- **No Manipuri reader has reviewed this output yet.**
- **Loud noise:** Shruti models sometimes repeat a phrase over and over in loud noise (measured on Shruti Hindi-family models, not on Manipuri).
- **Not tested on Manipuri:** noise, music, far-away or speaker-played audio, Linux, Windows and iOS. No punctuation; numbers come out as words.
