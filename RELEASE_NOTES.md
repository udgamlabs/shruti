# Release notes

## 1.7.0: Shruti Kannada joins the family (6 November 2026)

- **Shruti Kannada** (`shruti-kannada.cact`, 29.7 MB, [udgamlabs/shruti-kannada](https://huggingface.co/udgamlabs/shruti-kannada)):
  Kannada and English speech to text, English words in English. Results in `docs/kannada.md`.
- Examples: `transcribe.py --model kannada`, `SHRUTI_MODEL=kannada ./transcribe.sh`, and `?model=kannada` in the browser
  example. `SHA256SUMS` lists the Kannada file.

## 1.6.0: Shruti Odia joins the family (6 November 2026)

- **Shruti Odia** (`shruti-odia.cact`, 29.7 MB, [udgamlabs/shruti-odia](https://huggingface.co/udgamlabs/shruti-odia)):
  Odia and English speech to text, English words in English. Results in `docs/odia.md`.
- Examples: `transcribe.py --model odia`, `SHRUTI_MODEL=odia ./transcribe.sh`, and `?model=odia` in the browser example.
  `SHA256SUMS` lists the Odia file.

## 1.5.0: Shruti Gujarati joins the family (4 November 2026)

- **Shruti Gujarati** (`shruti-gujarati.cact`, 29.7 MB, [udgamlabs/shruti-gujarati](https://huggingface.co/udgamlabs/shruti-gujarati)):
  Gujarati and English speech to text, English words in English. Results in `docs/gujarati.md`.
- Examples: `transcribe.py --model gujarati`, `SHRUTI_MODEL=gujarati ./transcribe.sh`, and `?model=gujarati` in the browser
  example. `SHA256SUMS` lists the Gujarati file.

## 1.4.0: Shruti Telugu joins the family (4 November 2026)

- **Shruti Telugu** (`shruti-telugu.cact`, 29.7 MB, [udgamlabs/shruti-telugu](https://huggingface.co/udgamlabs/shruti-telugu)):
  Telugu and English speech to text, English words in English. Results in `docs/telugu.md`.
- Examples: `transcribe.py --model telugu`, `SHRUTI_MODEL=telugu ./transcribe.sh`, and `?model=telugu` in the browser
  example. `SHA256SUMS` lists the Telugu file.

## 1.3.0: Shruti Bengali joins the family (3 November 2026)

- **Shruti Bengali** (`shruti-bengali.cact`, 29.7 MB, [udgamlabs/shruti-bengali](https://huggingface.co/udgamlabs/shruti-bengali)):
  Bengali and English speech to text, English words in English. Results in `docs/bengali.md`.
- Examples: `transcribe.py --model bengali`, `SHRUTI_MODEL=bengali ./transcribe.sh`, and `?model=bengali` in the browser
  example. `SHA256SUMS` lists the Bengali file.

## 1.2.0: Shruti Tamil joins the family (3 November 2026)

- **Shruti Tamil** (`shruti-tamil.cact`, 29.7 MB, [udgamlabs/shruti-tamil](https://huggingface.co/udgamlabs/shruti-tamil)):
  Tamil and English speech to text, English words in English. Results in `docs/tamil.md`.
- Examples: `transcribe.py --model tamil`, `SHRUTI_MODEL=tamil ./transcribe.sh`, and `?model=tamil` in the browser
  example. `SHA256SUMS` lists the Tamil file.

## 1.1.0: Shruti Marathi joins the family (2 November 2026)

- **Shruti Marathi** (`shruti-marathi.cact`, 29.7 MB, [udgamlabs/shruti-marathi](https://huggingface.co/udgamlabs/shruti-marathi)):
  Marathi and English speech to text, English words in English. Results in `docs/marathi.md`.
- The Hindi model is now called **Shruti Hindi**. Nothing moved: its file, checksum and links are unchanged.
- Examples: `transcribe.py --model marathi`, `SHRUTI_MODEL=marathi ./transcribe.sh`, and `?model=marathi` in the
  browser example. `SHA256SUMS` lists the Marathi file.

## Documentation update (19 October 2026)

The model files and examples are unchanged (still 1.0.0; checksums in `SHA256SUMS` unchanged).

- **Whisper large-v3, same harness:** added to the README results table and `docs/comparison.md`, with a same-clip table
  and paired-bootstrap intervals. Shruti is 100× smaller than OpenAI's Whisper large-v3 and more accurate on Hindi on
  the same clips (Monsoon 20.46 vs 33.29, FLEURS 20.14 vs 28.88, Lahaja 22.87 vs 31.19 on the first 300 clips). Whisper
  is far better on English (Svarah 7.39 vs 21.35, LibriSpeech 2.25 vs 5.76 on the first 300 clips each); this is now
  listed under "Where it falls short".
- **Technical report:** linked from the README, with a BibTeX entry:
  [udgamlabs.com/research/shruti](https://udgamlabs.com/research/shruti).
- **Training:** "How it was trained" now gives a high-level description, and the data credits list every source with
  its licence.
- **Corrections:** Whisper-small's Monsoon and FLEURS numbers are given to two decimals (128.25, 59.42), as in
  `docs/comparison.md`; Whisper-Hindi2Hinglish-Swift's FLEURS, Lahaja and MUCS cells read "n/c" (not comparable, Roman
  script only) instead of "–"; the silence test is described exactly (50 clips of digital silence and 50 of very faint
  hiss); the results note points to the decoding settings in `docs/comparison.md` instead of saying "default decoding".

## 1.0.0 (19 October 2026)

First public release of Shruti, speech-to-text for Hindi, English and Hinglish. The weights are on Hugging Face
([udgamlabs/Shruti](https://huggingface.co/udgamlabs/Shruti)); this repository adds runnable examples, the full
comparison and checksums.

- **Model:** 55M parameters, built on Cactus Whistle (Apache-2.0) with all weights retrained on 2,732 hours of openly
  licensed Indian speech and 714 added Devanagari vocabulary pieces. Hindi in Devanagari, English words in Latin script.
- **Files:** `shruti.cact` (29.5 MB, 4-bit weights, for the Cactus needle engine) and `shruti.safetensors` (222 MB,
  full-precision weights). Checksums in `SHA256SUMS`.
- **Runs offline:** Python (`cactus-needle`), the `needle` command line, and the browser (WebAssembly).
- **Results** (word error rate, same harness for every model): MUCS Hinglish 35.21, the lowest of the open models we
  measured; Monsoon Hindi 20.46; LibriSpeech 5.95; 0 words invented on 100 silent clips. See `docs/comparison.md`.
- **Examples:** `examples/transcribe.py`, `examples/transcribe.sh` and the `examples/browser/` page.

Licence: Apache-2.0. Known limitations are listed in the README (pure Hindi behind larger Hindi-only models; Indian
English behind a 10× larger model; unseen names spelt by sound; noise and music not tested).
