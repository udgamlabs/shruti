"""Transcribe a file with Shruti in ONNX Runtime.

    pip install shruti-onnx
    python transcribe.py clip.wav --lang urdu
    python transcribe.py clip.wav --lang urdu --speech english      # all-English speech, especially with an Indian accent
    python transcribe.py clip.wav --lang urdu --low-memory          # int8 build: 53 MB download, about 120 MB of memory

The model downloads once and is then cached, so later runs work offline. WAV works out of the box;
`pip install soundfile` adds other formats.
"""
import argparse

from shruti_onnx import Shruti

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("audio")
parser.add_argument("--lang", required=True, help='which model: a word or a code, such as "urdu" or "ur"')
parser.add_argument("--speech", help='force the language of the speech: the model\'s own or "english" (default: detect)')
parser.add_argument("--low-memory", action="store_true")
parser.add_argument("--model", help="a local folder instead of the Hugging Face repo")
args = parser.parse_args()

options = {"variant": "int8" if args.low_memory else "4bit"}
if args.model:
    options["model"] = args.model
asr = Shruti.load(args.lang, **options)
result = asr.transcribe(args.audio, language=args.speech)
print(result["text"])
print(f"[language: {result['language']}]")
