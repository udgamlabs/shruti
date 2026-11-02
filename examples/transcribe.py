#!/usr/bin/env python3
"""Transcribe a WAV file with a Shruti model, offline, on this computer.

    pip install cactus-needle
    python transcribe.py clip.wav                     # Shruti Hindi; language detected automatically
    python transcribe.py clip.wav --model marathi     # Shruti Marathi (Marathi and English)
    python transcribe.py english.wav --language en    # only for speech that is all English

The first run downloads the model file (shruti.cact, 29.5 MB, from udgamlabs/Shruti; or shruti-marathi.cact, 29.7 MB,
from udgamlabs/shruti-marathi) from Hugging Face and the cactus-needle package
downloads Cactus's engine library (needle3); both are cached. After that it runs with no network.
Any WAV works (integer or float samples, any channel count); audio that is not 16 kHz needs `pip install
"cactus-needle[mic]"` for resampling.
"""
import os

# The cactus-needle Python package sends anonymous usage counts (function name, versions, OS) by default.
# This example turns that off before the package is imported. Delete this line to leave it on.
os.environ.setdefault("NEEDLE_TELEMETRY", "0")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402  (installed with cactus-needle)
from needle import Whistle  # noqa: E402

# One model per language (see SHA256SUMS): Hugging Face repo, file, sha256, its language tag.
MODELS = {
    "hindi": ("udgamlabs/Shruti", "shruti.cact", "b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065", "hi"),
    "marathi": ("udgamlabs/shruti-marathi", "shruti-marathi.cact",
                "f51a92944cc25515e703518cfef5ec9c52bf0db4cf91741530ebdded1df5bb05", "mr"),
}


def model_path(name: str) -> str:
    REPO, FILE, SHA256, _ = MODELS[name]
    path = hf_hub_download(REPO, FILE)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        sys.exit(f"{path}: sha256 {digest.hexdigest()} is not the released file ({SHA256}). "
                 "The file on Hugging Face has changed; check the release notes.")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline speech to text with Shruti Hindi or Shruti Marathi.")
    parser.add_argument("wav", help="a WAV file")
    parser.add_argument("--model", choices=sorted(MODELS), default="hindi",
                        help="hindi: Hindi, English and Hinglish (default); marathi: Marathi and English")
    parser.add_argument("--language", choices=["hi", "mr", "en"], default=None,
                        help="leave unset to detect it; 'en' helps only for speech that is entirely English")
    parser.add_argument("--keywords", default=None,
                        help="comma-separated names or terms to favour, e.g. 'Shruti, Udgam Labs'")
    args = parser.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] if args.keywords else None
    if args.language not in (None, "en", MODELS[args.model][3]):
        parser.error(f"--language {args.language} does not match --model {args.model}")
    model = Whistle(model_path(args.model))
    result = model.transcribe(args.wav, language=args.language, keywords=keywords)
    print(result["text"])
    print(f"language: {result['language']}")


if __name__ == "__main__":
    main()
