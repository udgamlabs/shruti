# shruti-onnx

Shruti: offline speech-to-text, one small model per language. Each model covers its language, English, and the two mixed. Models and full results: [udgamlabs/Shruti-ONNX](https://huggingface.co/udgamlabs/Shruti-ONNX).

Languages: Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu. Use the word or the code.

```bash
pip install shruti-onnx
```

```python
from shruti_onnx import Shruti

asr = Shruti.load("urdu")                    # or "ur"; variant="int8" for low memory on ARM
print(asr.transcribe("clip.wav")["text"])    # language detected; language="en" to set English
```

Command line: `shruti-onnx clip.wav --lang urdu`. `loops="cut"` removes a repeat and what follows it.

## Where it falls short

Nine of these languages (Dogri, Hindi, Kannada, Konkani, Odia, Punjabi, Sindhi, Tamil, Telugu) failed our written comparison against the original engine, and none met every rule. The model card names each miss and its size. On Linux x86, use the default 4-bit build: the int8 and sherpa-onnx builds were less accurate there on the machine we tested. A model can get stuck repeating a phrase; the text stays as the model wrote it unless you ask to cut it. The 4-bit build needs about 1 GB of memory once loaded in ONNX Runtime.

Apache-2.0. Built on [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle).

**About Udgam Labs.** Udgam Labs is an independent AI research lab. [udgamlabs.com](https://udgamlabs.com)
