#!/usr/bin/env python3
"""Transcribe a WAV file with a Shruti model, offline, on this computer.

    pip install cactus-compute
    python transcribe.py clip.wav                     # Shruti Hindi; language detected automatically
    python transcribe.py clip.wav --model marathi     # Shruti Marathi (Marathi and English)
    python transcribe.py clip.wav --model tamil       # Shruti Tamil (Tamil and English)
    python transcribe.py clip.wav --model bengali     # Shruti Bengali (Bengali and English)
    python transcribe.py clip.wav --model telugu      # Shruti Telugu (Telugu and English)
    python transcribe.py clip.wav --model gujarati    # Shruti Gujarati (Gujarati and English)
    python transcribe.py clip.wav --model odia        # Shruti Odia (Odia and English)
    python transcribe.py clip.wav --model kannada     # Shruti Kannada (Kannada and English)
    python transcribe.py clip.wav --model urdu        # Shruti Urdu (Urdu and English)
    python transcribe.py clip.wav --model malayalam # Shruti Malayalam (Malayalam and English)
    python transcribe.py clip.wav --model assamese  # Shruti Assamese (Assamese and English)
    python transcribe.py clip.wav --model sanskrit  # Shruti Sanskrit (spoken Sanskrit and English)
    python transcribe.py clip.wav --model punjabi  # Shruti Punjabi (Punjabi and English)
    python transcribe.py clip.wav --model bodo  # Shruti Bodo (Bodo and English)
    python transcribe.py clip.wav --model dogri  # Shruti Dogri (Dogri and English)
    python transcribe.py clip.wav --model kashmiri  # Shruti Kashmiri (Kashmiri and English)
    python transcribe.py clip.wav --model konkani  # Shruti Konkani (Konkani and English)
    python transcribe.py clip.wav --model maithili  # Shruti Maithili (Maithili and English)
    python transcribe.py clip.wav --model manipuri  # Shruti Manipuri (Manipuri and English)
    python transcribe.py clip.wav --model nepali  # Shruti Nepali (Nepali and English)
    python transcribe.py clip.wav --model santali  # Shruti Santali (Santali and English)
    python transcribe.py clip.wav --model sindhi  # Shruti Sindhi (Sindhi and English)
    python transcribe.py english.wav --language en    # only for speech that is all English

The first run downloads the model file (shruti.cact, 29.5 MB, from udgamlabs/Shruti; or shruti-marathi.cact, 29.7 MB,
from udgamlabs/shruti-marathi) from Hugging Face and the cactus-compute package
downloads Cactus's engine library; both are cached. After that it runs with no network.
Any WAV works (integer or float samples, any channel count); audio that is not 16 kHz needs `pip install
"cactus-compute[mic]"` for resampling.
"""
import os

# The cactus-compute Python package sends anonymous usage counts (function name, versions, OS) by default.
# This example turns that off before the package is imported. Delete this line to leave it on.
os.environ.setdefault("CACTUS_TELEMETRY", "0")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402  (installed with cactus-compute)
from cactus import Whistle  # noqa: E402

# One model per language (see SHA256SUMS): Hugging Face repo, file, sha256, its language tag.
MODELS = {
    "hindi": ("udgamlabs/Shruti", "shruti.cact", "b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065", "hi"),
    "marathi": ("udgamlabs/shruti-marathi", "shruti-marathi.cact",
                "f51a92944cc25515e703518cfef5ec9c52bf0db4cf91741530ebdded1df5bb05", "mr"),
    "tamil": ("udgamlabs/shruti-tamil", "shruti-tamil.cact",
              "8202f454350fb9ca2d2115ef5e81332c99de93b7c2f7c06fc9d2457995a4dec8", "ta"),
    "bengali": ("udgamlabs/shruti-bengali", "shruti-bengali.cact",
                "0a423c25f534e2099b43bc4e826018978c44a9f523cea19125686883a2a7c572", "bn"),
    "telugu": ("udgamlabs/shruti-telugu", "shruti-telugu.cact",
               "197166720f32163bc2102ebf9dc0374d51ec9928175649071da92d943cca0d9e", "te"),
    "gujarati": ("udgamlabs/shruti-gujarati", "shruti-gujarati.cact",
                 "23fd52c8edb36b24383c70b82d6dd21bfa7b070ca1606b15510e75bbf3822e1f", "gu"),
    "odia": ("udgamlabs/shruti-odia", "shruti-odia.cact",
             "e2818089abbd521b2224c01eec0d40e926673427d1abb0f78c27ebcfef71541d", "or"),
    "kannada": ("udgamlabs/shruti-kannada", "shruti-kannada.cact",
                "492cdfdea934061edfefd14f2bd1e0255468adc8d0f0e3e9c2847cee69d5d5d8", "kn"),
    "urdu": ("udgamlabs/shruti-urdu", "shruti-urdu.cact",
             "b5aaecd79cad932f24b1ab36ba61973d010bb8b14c1e8ee45ae9ad66955b4e8a", "ur"),
    "malayalam": ("udgamlabs/shruti-malayalam", "shruti-malayalam.cact",
                 "5c8e3904c6fcbe984a26140835026ebb30f0df29596860ca5b9dabdc64513372", "ml"),
    "assamese": ("udgamlabs/shruti-assamese", "shruti-assamese.cact",
                "86b41d774be36b6eadc4fb0ae906a8f444705b822f5c433e68e49ff3de56cd0e", "as"),
    "sanskrit": ("udgamlabs/shruti-sanskrit", "shruti-sanskrit.cact",
            "d6a0efbad1000edadad4334b1263bf860f8ef845c36ff2c81d83a9f8e8ad507c", "sa"),
    "punjabi": ("udgamlabs/shruti-punjabi", "shruti-punjabi.cact",
            "30d8f09ceac37a0d009f480896e3daac123dde94d1e2e913a0983ecdcde7fa36", "pa"),
    "bodo": ("udgamlabs/shruti-bodo", "shruti-bodo.cact",
            "fc38949cb343a7f914988f920286c6ee24d8166fafaad4ee4784af638b089580", "brx"),
    "dogri": ("udgamlabs/shruti-dogri", "shruti-dogri.cact",
            "5f0fc33adfc8bd1e2b0c973feaa34bf43b1c6f552ff2d0be300353d7339591bd", "doi"),
    "kashmiri": ("udgamlabs/shruti-kashmiri", "shruti-kashmiri.cact",
            "142cab90aba96992d3a0f5913ff72ba0e2097edb3aad613eeae247ff0cbc3504", "ks"),
    "konkani": ("udgamlabs/shruti-konkani", "shruti-konkani.cact",
            "c6c4de2cbc4c7e45af03cca684f4ed2c98c83ac5d68fa6044923a38b4ce902ff", "kok"),
    "maithili": ("udgamlabs/shruti-maithili", "shruti-maithili.cact",
            "7f87b9f2ce702956ad49bcec0f1da999bf4d38b3610b3d354ee9fac93f084842", "mai"),
    "manipuri": ("udgamlabs/shruti-manipuri", "shruti-manipuri.cact",
            "e3152c6b3ff1214ec9492811e51cd6e43037efdfb3d935df6238aca897975e5d", "mni"),
    "nepali": ("udgamlabs/shruti-nepali", "shruti-nepali.cact",
            "af3bf553a1448a468a376e491eedfaf5f53ee16ed9c6855173baa3d6c21aa7a0", "ne"),
    "santali": ("udgamlabs/shruti-santali", "shruti-santali.cact",
            "1faad55924ed6c01c3cff970c919b8c82c3ecc3712da7a22ce51f6344a125699", "sat"),
    "sindhi": ("udgamlabs/shruti-sindhi", "shruti-sindhi.cact",
            "d58f4646ca8fa87d62b0cea6ae53ba7c66fda6bd04ce18e2e37479ae72a79294", "sd"),
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
    parser = argparse.ArgumentParser(description="Offline speech to text with a Shruti model, one small model per language, each with English.")
    parser.add_argument("wav", help="a WAV file")
    parser.add_argument("--model", choices=sorted(MODELS), default="hindi",
                        help="hindi: Hindi, English and Hinglish (default); each other model: that language and English")
    parser.add_argument("--language", choices=["hi", "mr", "ta", "bn", "te", "gu", "or", "kn", "ur", "ml", "as", "sa", "pa", "brx", "doi", "ks", "kok", "mai", "mni", "ne", "sat", "sd", "en"], default=None,
                        help="leave unset to detect it; 'en' helps only for speech that is entirely English; set 'ne' for Nepali (docs/nepali.md)")
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
