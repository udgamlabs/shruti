// Transcribe a WAV file with Shruti in Node (onnxruntime-web, WebAssembly).
//
//   npm install
//   node transcribe.mjs clip.wav urdu                  // a word or a code; the speech language is detected
//   node transcribe.mjs clip.wav urdu english          // all-English speech, especially with an Indian accent
//
// SHRUTI_MODEL_DIR points at a local copy of one model folder (<...>/ur/onnx/) instead of downloading it.
import fs from "node:fs";
import * as ort from "onnxruntime-web";
import { createShruti, decodeWav } from "shruti-onnx";

const [file, language, speech = null] = process.argv.slice(2);
if (!file || !language) {
  console.error("usage: node transcribe.mjs clip.wav <language> [english]");
  process.exit(2);
}

ort.env.wasm.numThreads = 1;
const baseUrl = process.env.SHRUTI_MODEL_DIR ?? null;
const asr = await createShruti({ ort, language, baseUrl });

const bytes = fs.readFileSync(file);
const samples = decodeWav(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
const { text, language: detected } = await asr.transcribe(samples, { language: speech });
console.log(text);
console.log(`[language: ${detected}]`);
