"""Shruti: offline speech-to-text, one language and English in one small model, in ONNX Runtime.

    from shruti_onnx import Shruti
    asr = Shruti.load("tamil")                # a plain word or a code ("ta"); see LANGUAGES
    print(asr.transcribe("clip.wav")["text"])

    Shruti.load("ta", variant="int8")         # low memory; the default is the 4-bit build, the smallest download
    asr.transcribe(samples_16k, language="en")   # None (the default) detects: the model's language or English

Decoding is 5-beam search with length-normalised scores, which reproduces the Cactus engine's transcripts. The
search is the same as the research runner udgam-voice/asr/onnx/shruti_onnx.py (tested equal in tests/).
"""
import json
import os
import wave

import numpy as np
import onnxruntime as ort

__version__ = "0.2.0"
DEFAULT_REPO = "udgamlabs/Shruti-ONNX"
BASE_TAGS = ("en", "de", "fr", "es", "it", "nl", "pl")  # Cactus Whistle's tags; a Shruti model adds its own language
VARIANTS = {"4bit": "onnx", "int8": "onnx-int8"}
_ALL_LANGUAGES = {"as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "ks": "Kashmiri",
             "kok": "Konkani", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri", "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
             "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu"}
RELEASED = ("as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur")  # set by set_release.py
LANGUAGES = {c: _ALL_LANGUAGES[c] for c in RELEASED}
_ALIASES = {"bangla": "bn", "oriya": "or", "meitei": "mni", "meetei": "mni", "panjabi": "pa"}
LOOPS_DEFAULT = "off"
RATE = 16000


def language_code(name):
    """"urdu", "Urdu" or "ur" -> "ur". Raises ValueError listing the languages."""
    key = str(name).strip().lower()
    if key in _ALL_LANGUAGES:
        return key
    for code, word in _ALL_LANGUAGES.items():
        if key == word.lower():
            return code
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"unknown language {name!r}; choose one of: " + ", ".join(f"{w} ({c})" for c, w in sorted(LANGUAGES.items(), key=lambda x: x[1])))


def read_audio(path):
    """16 kHz mono float32 samples from a file: soundfile when installed, else 16/24/32-bit PCM WAV via the standard
    library. Other sample rates are resampled (soxr when installed, else linear interpolation)."""
    try:
        import soundfile as sf
        x, rate = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(1)
    except ImportError:
        with wave.open(path, "rb") as w:
            ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
        if width == 3:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
            x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int8).astype(np.int32) << 16))
            x = x / 8388608.0
        else:
            x = np.frombuffer(raw, {1: np.uint8, 2: np.int16, 4: np.int32}[width]).astype(np.float64)
            x = (x - 128) / 128.0 if width == 1 else x / float(2 ** (8 * width - 1))
        x = x.reshape(-1, ch).mean(1).astype(np.float32)
    return resample(x, rate)


def resample(x, rate):
    if rate == RATE:
        return np.asarray(x, np.float32)
    try:
        import soxr
        return soxr.resample(x, rate, RATE).astype(np.float32)
    except ImportError:
        n = int(round(len(x) * RATE / rate))
        return np.interp(np.arange(n) * rate / RATE, np.arange(len(x)), x).astype(np.float32)


class Shruti:
    """Runs the three graphs: frontend (audio -> features), encoder (features -> audio memory), decoder (one token
    at a time, with its state carried between calls)."""

    def __init__(self, folder, threads=2):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        load = lambda f: ort.InferenceSession(os.path.join(folder, f), so, providers=["CPUExecutionProvider"])  # noqa
        self.frontend, self.encoder, self.decoder = load("frontend.onnx"), load("encoder.onnx"), load("decoder.onnx")
        meta = json.load(open(os.path.join(folder, "shruti.json"), encoding="utf-8"))
        self.vocab, self.cfg = meta["vocab"], meta["config"]
        self.text, self.bos, self.eos = self.cfg["text"], self.cfg["bos"], self.cfg["eos"]
        self.tags = {lang: self.text + i for i, lang in enumerate(self.cfg["languages"])}
        own = [x for x in self.cfg["languages"] if x not in BASE_TAGS]
        self.languages = (own[0], "en")  # the model's own language and English: what it was trained on
        self.state_names = [i.name for i in self.decoder.get_inputs()][2:]
        shapes = {i.name: i.shape for i in self.decoder.get_inputs()}
        self.state_shapes = {n: [d if isinstance(d, int) else 0 for d in shapes[n]] for n in self.state_names}

    @classmethod
    def load(cls, language, variant="4bit", model=DEFAULT_REPO, threads=2):
        """language: a plain word or a code ("urdu" or "ur"); see LANGUAGES.
        variant: "4bit" (32 MB, the smallest download) or "int8" (53 MB, about 120 MB of memory instead of about 1 GB).
        model: a Hugging Face repo id, or a local folder holding <code>/onnx/ and <code>/onnx-int8/."""
        code = language_code(language)
        if code not in LANGUAGES and not os.path.isdir(model):
            raise ValueError(f"{_ALL_LANGUAGES[code]} is not in this release; choose one of: " + ", ".join(f"{w} ({c})" for c, w in sorted(LANGUAGES.items(), key=lambda x: x[1])))
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}, not {variant!r}")
        rel = f"{code}/{VARIANTS[variant]}"
        if os.path.isdir(model):
            path = model
        else:
            from huggingface_hub import snapshot_download
            path = snapshot_download(model, allow_patterns=[f"{rel}/*"])
        if not os.path.isdir(os.path.join(path, rel)):
            raise FileNotFoundError(f"no {rel} folder in {model}: this language is not in this release; see its table of languages")
        asr = cls(os.path.join(path, rel), threads)
        if asr.languages[0] != code:
            raise ValueError(f"{rel} holds a {asr.languages[0]!r} model, not {code!r}")
        return asr

    def detok(self, ids):
        """Pieces to text (sentencepiece rules: a leading ▁ is a space, <0xNN> pieces are raw UTF-8 bytes)."""
        out, raw = [], bytearray()
        for i in ids:
            p = self.vocab[i]
            if len(p) == 6 and p.startswith("<0x") and p.endswith(">"):
                raw.append(int(p[3:5], 16))
                continue
            if raw:
                out.append(raw.decode("utf-8", "ignore"))
                raw.clear()
            if not (p.startswith("<") and p.endswith(">")):
                out.append(p)
        if raw:
            out.append(raw.decode("utf-8", "ignore"))
        return "".join(out).replace("▁", " ").strip()

    def init_state(self, B):
        st = {}
        for n in self.state_names:
            shp = list(self.state_shapes[n])
            if n == "pos":
                st[n] = np.zeros(1, np.int64)
                continue
            shp[0] = B
            st[n] = np.zeros(shp, np.int64 if n == "hist" else np.float32)
        return st

    def step(self, tokens, cross, st):
        r = self.decoder.run(None, {"tokens": tokens, "cross": cross, **st})
        return r[0][:, 0], dict(zip(self.state_names, r[1:]))

    def encode(self, wav):
        feats = self.frontend.run(None, {"audio": wav[None].astype(np.float32)})[0]
        n = np.array([len(wav) // self.cfg["hop"]], np.int64)
        return self.encoder.run(None, {"features": feats, "frames": n})[0]

    @staticmethod
    def logsoftmax(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))

    def transcribe(self, audio, language=None, beams=5, numbers="words", window_s=30.0, length_norm=True,
                   tok_per_s=12.0, loops=LOOPS_DEFAULT):
        """audio: a file path, or 16 kHz mono float32 samples in [-1, 1].
        language: None detects between the model's own language and English; or name either one ("tamil", "ta", "english", "en").
        beams: 5 (the engine's search) or 1 (greedy, faster).
        numbers: "words" (as spoken: "ढाई सौ रुपये") or "digits" ("₹250"; Hindi model only, needs the
        udgam_textnorm normaliser, not on PyPI yet).
        loops: "off" keeps the model's output as it is; "cut" keeps the first copy of a repeated pattern and drops
        everything after it.
        Up to 30 s is one pass; longer audio is cut into equal windows of at most 30 s.
        -> {"text", "language", "tokens"}"""
        if language is not None:
            language = "en" if str(language).strip().lower() in ("en", "english") else language_code(language)
            if language not in self.languages:
                raise ValueError(f"this is the {LANGUAGES[self.languages[0]]} model: language must be {self.languages[0]!r}, 'en' or None to detect, not {language!r}")
        if numbers == "digits" and self.languages[0] != "hi":
            raise ValueError('numbers="digits" is tested for the Hindi model only')
        if loops not in ("off", "cut"):
            raise ValueError('loops must be "off" or "cut"')
        wav = read_audio(audio) if isinstance(audio, (str, os.PathLike)) else np.asarray(audio, np.float32)
        if len(wav) == 0 or np.abs(wav).max() < 1e-4:  # digital silence -> empty, as the Cactus engine returns
            return {"text": "", "language": language, "tokens": []}
        n = max(1, int(np.ceil(len(wav) / (window_s * RATE))))
        parts = [self._transcribe(w, language, beams, tok_per_s, length_norm, loops) for w in np.array_split(wav, n)]
        text = " ".join(p["text"] for p in parts if p["text"])
        lang = parts[0]["language"]
        if numbers == "digits":
            try:
                from udgam_textnorm import denormalize
            except ImportError as e:
                raise ImportError('numbers="digits" needs the udgam_textnorm number normaliser, which is not on PyPI yet') from e
            text = denormalize(text, lang, style="digits")
        elif numbers != "words":
            raise ValueError('numbers must be "words" or "digits"')
        return {"text": text, "language": lang, "tokens": [t for p in parts for t in p["tokens"]]}

    @staticmethod
    def cut_loops(ids, max_period=8, repeats=3):
        """Cut a token loop (a pattern of up to 8 tokens repeated 3+ times back to back): keep its first copy."""
        n = len(ids)
        for start in range(n):
            for k in range(1, max_period + 1):
                if start + k * repeats > n:
                    break
                pat = ids[start:start + k]
                if all(ids[start + r * k:start + (r + 1) * k] == pat for r in range(1, repeats)):
                    return ids[:start + k]
        return ids

    def _transcribe(self, wav, language, beams, tok_per_s, length_norm, loops):
        tidy = self.cut_loops if loops == "cut" else (lambda ids: ids)
        cross = self.encode(wav)
        cap = int(min(self.cfg["max_tokens"] - 2, len(wav) / RATE * tok_per_s + 8))
        st = self.init_state(1)
        if language is None:
            lg, st = self.step(np.array([[self.bos]], np.int64), cross, st)
            language = max(self.languages, key=lambda lang: lg[0, self.tags[lang]])
            lg, st = self.step(np.array([[self.tags[language]]], np.int64), cross, st)
        else:
            lg, st = self.step(np.array([[self.bos, self.tags[language]]], np.int64), cross, st)
        lg[:, self.text:] = -np.inf
        if beams <= 1:
            ids = []
            for _ in range(cap):
                k = int(lg[0].argmax())
                if k == self.eos:
                    break
                ids.append(k)
                lg, st = self.step(np.array([[k]], np.int64), cross, st)
                lg[:, self.text:] = -np.inf
            ids = tidy(ids)
            return {"text": self.detok(ids), "language": language, "tokens": ids}
        norm = (lambda sc, s: sc / (len(s) + 1)) if length_norm else (lambda sc, s: sc)
        lp = self.logsoftmax(lg[0])
        first = np.argsort(-lp)[:beams]
        seqs = [[int(k)] for k in first]
        scores = lp[first].astype(np.float64)
        finished = [(scores[i], s[:-1]) for i, s in enumerate(seqs) if s[-1] == self.eos]
        live = [i for i, s in enumerate(seqs) if s[-1] != self.eos]
        seqs, scores = [seqs[i] for i in live], scores[live]
        st = {n: (v if n == "pos" else np.repeat(v, len(seqs), 0)) for n, v in st.items()}
        for _ in range(cap - 1):
            if not seqs:
                break
            lg, st = self.step(np.array([[s[-1]] for s in seqs], np.int64), cross, st)
            lg[:, self.text:] = -np.inf
            cand = scores[:, None] + self.logsoftmax(lg)
            new_seqs, new_scores, src = [], [], []
            for flat in np.argsort(-cand, axis=None)[: 2 * beams]:
                b, k = divmod(int(flat), cand.shape[1])
                if k == self.eos:
                    finished.append((cand[b, k], seqs[b]))
                else:
                    new_seqs.append(seqs[b] + [k])
                    new_scores.append(cand[b, k])
                    src.append(b)
                if len(new_seqs) == beams:
                    break
            best_done = max((norm(sc, s) for sc, s in finished), default=-np.inf)
            if len(finished) >= beams and (not new_scores or
                                           max(norm(sc, s) for sc, s in zip(new_scores, new_seqs)) < best_done):
                break
            seqs, scores = new_seqs, np.array(new_scores)
            st = {n: (v if n == "pos" else v[src]) for n, v in st.items()}
        if not finished:
            finished = list(zip(scores, seqs))
        best = tidy(max(finished, key=lambda f: norm(*f))[1])
        return {"text": self.detok(best), "language": language, "tokens": best}


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="shruti-onnx", description="Offline speech-to-text: one language and English in one small model.")
    p.add_argument("audio", nargs="+", help="audio files (WAV; other formats with `pip install soundfile`)")
    p.add_argument("--lang", required=True, help="which model: a plain word or a code, e.g. tamil or ta")
    p.add_argument("--int8", action="store_true", help="the low-memory build (53 MB download, ~120 MB of memory)")
    p.add_argument("--model", default=DEFAULT_REPO, help="Hugging Face repo id or local folder")
    p.add_argument("--force-language", help="skip detection: the model's own language or en")
    p.add_argument("--numbers", choices=["words", "digits"], default="words")
    p.add_argument("--greedy", action="store_true", help="faster, slightly less accurate")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args(argv)
    try:
        asr = Shruti.load(a.lang, "int8" if a.int8 else "4bit", a.model, a.threads)
    except ValueError as e:
        p.error(str(e))
    for f in a.audio:
        r = asr.transcribe(f, language=a.force_language, beams=1 if a.greedy else 5, numbers=a.numbers)
        print(f"{f}\t{r['language']}\t{r['text']}")


if __name__ == "__main__":
    main()
