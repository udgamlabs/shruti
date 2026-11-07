// Shruti: offline speech-to-text, one language and English in one small model, in the browser (onnxruntime-web) or Node.
//
//   import * as ort from "onnxruntime-web";
//   import { createShruti } from "shruti-onnx";
//   const asr = await createShruti({ ort, language: "urdu" });   // a plain word or a code ("ur"); see LANGUAGES
//   const { text } = await asr.transcribe(samples16k);            // Float32Array, 16 kHz mono
//
//   createShruti({ ort, language: "ur", variant: "int8" })        // low memory; the default is the 4-bit build
//   createShruti({ ort, baseUrl: "/models/ur/onnx/" })            // your own copy: a URL, or a folder path in Node
//   asr.transcribe(samples16k, { language: "en" })                // null (the default) detects: the model's language or English
//
// Decoding is 5-beam search with length-normalised scores, the same search as the Python package (shruti_onnx).

export const RATE = 16000;
const LOOPS_DEFAULT = "off";
const REPO = "https://huggingface.co/udgamlabs/Shruti-ONNX/resolve/main/";
const VARIANTS = { "4bit": "onnx", int8: "onnx-int8" };
const ALL_LANGUAGES = {
  as: "Assamese", bn: "Bengali", brx: "Bodo", doi: "Dogri", gu: "Gujarati", hi: "Hindi", kn: "Kannada", ks: "Kashmiri", kok: "Konkani", mai: "Maithili",
  ml: "Malayalam", mni: "Manipuri", mr: "Marathi", ne: "Nepali", or: "Odia", pa: "Punjabi", sa: "Sanskrit", sat: "Santali", sd: "Sindhi", ta: "Tamil", te: "Telugu", ur: "Urdu",
};
const RELEASED = ["as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"]; // set by set_release.py
export const LANGUAGES = Object.fromEntries(RELEASED.map((c) => [c, ALL_LANGUAGES[c]]));
const ALIASES = { bangla: "bn", oriya: "or", meitei: "mni", meetei: "mni", panjabi: "pa" };

const released = () => Object.entries(LANGUAGES).sort((a, b) => a[1].localeCompare(b[1])).map(([c, w]) => `${w} (${c})`).join(", ");

// "tamil", "Tamil" or "ta" -> "ta". Throws, listing the languages.
export function languageCode(name) {
  const key = String(name).trim().toLowerCase();
  if (Object.hasOwn(ALL_LANGUAGES, key)) return key;
  for (const [code, word] of Object.entries(ALL_LANGUAGES)) if (word.toLowerCase() === key) return code;
  if (Object.hasOwn(ALIASES, key)) return ALIASES[key];
  throw new Error(`unknown language ${JSON.stringify(name)}; choose one of: ${released()}`);
}

// A URL through fetch; in Node, a plain path through the file system.
async function defaultFetch(path) {
  const remote = typeof process === "undefined" || !process.versions?.node || /^[a-z]+:\/\//i.test(String(path));
  if (remote) return fetch(path);
  const fs = await import("node:fs/promises");
  const b = await fs.readFile(path);
  return { ok: true, status: 200, arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.length), json: async () => JSON.parse(b.toString("utf8")) };
}

export async function createShruti({ ort, language = null, variant = "4bit", baseUrl = null, fetchFn = defaultFetch, sessionOptions = {} }) {
  if (!ort) throw new Error('pass the runtime: import * as ort from "onnxruntime-web"; createShruti({ ort, language: "urdu" })');
  const code = language === null ? null : languageCode(language);
  if (!(variant in VARIANTS)) throw new Error(`variant must be one of ${Object.keys(VARIANTS).join(", ")}`);
  if (baseUrl === null) {
    if (code === null) throw new Error('name a language: createShruti({ ort, language: "urdu" }), or pass baseUrl for your own copy');
    if (!Object.hasOwn(LANGUAGES, code)) throw new Error(`${ALL_LANGUAGES[code]} is not in this release; choose one of: ${released()}`);
    baseUrl = `${REPO}${code}/${VARIANTS[variant]}/`;
  } else if (!baseUrl.endsWith("/")) baseUrl += "/";
  const get = async (f) => {
    const r = await fetchFn(baseUrl + f);
    if (r.ok === false) throw new Error(`could not fetch ${baseUrl + f}: HTTP ${r.status}` + (r.status === 404 ? " (this language may not be in this release; see its table of languages)" : ""));
    return r;
  };
  const opt = { executionProviders: ["wasm"], graphOptimizationLevel: "all", ...sessionOptions };
  const load = async (f) => ort.InferenceSession.create(new Uint8Array(await (await get(f)).arrayBuffer()), opt);
  const [frontend, encoder, decoder, meta] = await Promise.all([
    load("frontend.onnx"), load("encoder.onnx"), load("decoder.onnx"), get("shruti.json").then((r) => r.json()),
  ]);
  const asr = new Shruti(ort, frontend, encoder, decoder, meta);
  if (code !== null && asr.languages[0] !== code) throw new Error(`${baseUrl} holds a ${asr.languages[0]} model, not ${code}`);
  return asr;
}

class Shruti {
  constructor(ort, frontend, encoder, decoder, meta) {
    Object.assign(this, { ort, frontend, encoder, decoder });
    this.cfg = meta.config;
    this.vocab = meta.vocab;
    this.text = this.cfg.text;
    this.bos = this.cfg.bos;
    this.eos = this.cfg.eos;
    this.tags = Object.fromEntries(this.cfg.languages.map((l, i) => [l, this.text + i]));
    // the model's own language and English: what it was trained on (the other tags are Cactus Whistle's)
    this.languages = [this.cfg.languages.find((l) => !["en", "de", "fr", "es", "it", "nl", "pl"].includes(l)), "en"];
    this.stateNames = decoder.inputNames.slice(2);
  }

  detok(ids) {
    let out = "", bytes = [];
    const flush = () => { if (bytes.length) { out += new TextDecoder().decode(new Uint8Array(bytes)); bytes = []; } };
    for (const i of ids) {
      const p = this.vocab[i];
      if (p.length === 6 && p.startsWith("<0x") && p.endsWith(">")) { bytes.push(parseInt(p.slice(3, 5), 16)); continue; }
      flush();
      if (!(p.startsWith("<") && p.endsWith(">"))) out += p;
    }
    flush();
    return out.replaceAll("▁", " ").trim();
  }

  initState(B) {
    const T = this.ort.Tensor, st = {};
    for (const n of this.stateNames) {
      if (n === "pos") st[n] = new T("int64", new BigInt64Array(1), [1]);
      else if (n === "hist") st[n] = new T("int64", new BigInt64Array(2 * B), [B, 2]);
      else if (/^tap\d/.test(n)) st[n] = new T("float32", new Float32Array(B * 2 * 608), [B, 2, 608]);
      else if (/^k\d/.test(n)) st[n] = new T("float32", new Float32Array(0), [B, 2, 0, this.cfg.qk_dim]);
      else if (/^v\d/.test(n)) st[n] = new T("float32", new Float32Array(0), [B, 2, 0, this.cfg.v_dim]);
      else st[n] = new T("float32", new Float32Array(B * 9 * 512), [B, 9, 512]);
    }
    return st;
  }

  // keep rows `src` of every state tensor (beam reordering); pos is shared
  pick(st, src) {
    const T = this.ort.Tensor, out = {};
    for (const [n, t] of Object.entries(st)) {
      if (n === "pos") { out[n] = t; continue; }
      const row = t.dims.slice(1).reduce((a, b) => a * b, 1);
      const data = new t.data.constructor(src.length * row);
      src.forEach((b, i) => data.set(t.data.subarray(b * row, (b + 1) * row), i * row));
      out[n] = new T(t.type, data, [src.length, ...t.dims.slice(1)]);
    }
    return out;
  }

  async step(tokens, cross, st) {
    const T = this.ort.Tensor;
    const r = await this.decoder.run({ tokens: new T("int64", BigInt64Array.from(tokens.flat(), BigInt), [tokens.length, tokens[0].length]), cross, ...st });
    const V = r.logits.dims[2], B = r.logits.dims[0], lg = [];
    for (let b = 0; b < B; b++) lg.push(Float32Array.from(r.logits.data.subarray(b * V, (b + 1) * V)));
    const next = {};
    for (const n of this.stateNames) next[n] = r["new_" + n];
    return [lg, next];
  }

  async encode(wav) {
    const T = this.ort.Tensor;
    const feats = (await this.frontend.run({ audio: new T("float32", wav, [1, wav.length]) })).features;
    const frames = new T("int64", BigInt64Array.from([BigInt(Math.floor(wav.length / this.cfg.hop))]), [1]);
    return (await this.encoder.run({ features: feats, frames })).cross;
  }

  static logSoftmax(x) {
    let m = -Infinity;
    for (const v of x) if (v > m) m = v;
    let s = 0;
    for (const v of x) s += Math.exp(v - m);
    const l = m + Math.log(s), out = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] - l;
    return out;
  }

  static cutLoops(ids, maxPeriod = 8, repeats = 3) {
    const n = ids.length;
    for (let s = 0; s < n; s++) {
      for (let k = 1; k <= maxPeriod; k++) {
        if (s + k * repeats > n) break;
        let ok = true;
        for (let r = 1; r < repeats && ok; r++) for (let j = 0; j < k; j++) if (ids[s + r * k + j] !== ids[s + j]) { ok = false; break; }
        if (ok) return ids.slice(0, s + k);
      }
    }
    return ids;
  }

  // samples: Float32Array, 16 kHz mono. Options: language: null (detect between the model's own language and English), or either one by
  // word or code ("tamil", "ta", "english", "en"), beams 5 | 1,
  // numbers "words" | "digits" (Hindi model only), loops "off" (the model's output as it is) | "cut" (keep the first copy of a
  // repeated pattern and drop everything after it).
  async transcribe(samples, { language = null, beams = 5, numbers = "words", windowS = 30, lengthNorm = true, tokPerS = 12, loops = LOOPS_DEFAULT } = {}) {
    if (language !== null) {
      const key = String(language).trim().toLowerCase();
      language = key === "en" || key === "english" ? "en" : languageCode(language);
      if (!this.languages.includes(language)) throw new Error(`this is the ${LANGUAGES[this.languages[0]]} model: language must be "${this.languages[0]}", "en" or null to detect, not "${language}"`);
    }
    if (numbers === "digits" && this.languages[0] !== "hi") throw new Error('numbers: "digits" is tested for the Hindi model only');
    if (loops !== "off" && loops !== "cut") throw new Error('loops must be "off" or "cut"');
    let peak = 0;
    for (const v of samples) peak = Math.max(peak, Math.abs(v));
    if (samples.length === 0 || peak < 1e-4) return { text: "", language, tokens: [] };  // digital silence
    const n = Math.max(1, Math.ceil(samples.length / (windowS * RATE)));
    const parts = [], size = Math.floor(samples.length / n), extra = samples.length % n;
    for (let i = 0, start = 0; i < n; i++) {  // equal windows, like numpy.array_split
      const len = size + (i < extra ? 1 : 0);
      parts.push(await this._transcribe(samples.subarray(start, start + len), language, beams, tokPerS, lengthNorm, loops));
      start += len;
    }
    let text = parts.map((p) => p.text).filter(Boolean).join(" ");
    const lang = parts[0].language;
    if (numbers === "digits") {
      let norm;
      try { norm = await import("udgam-textnorm"); } catch (e) { throw new Error('numbers: "digits" needs the udgam-textnorm package, which is not on npm yet'); }
      text = norm.denormalize(text, lang, "digits");
    } else if (numbers !== "words") throw new Error('numbers must be "words" or "digits"');
    return { text, language: lang, tokens: parts.flatMap((p) => p.tokens) };
  }

  async _transcribe(wav, language, beams, tokPerS, lengthNorm, loops) {
    const tidy = loops === "cut" ? (ids) => Shruti.cutLoops(ids) : (ids) => ids;
    const cross = await this.encode(wav);
    const cap = Math.floor(Math.min(this.cfg.max_tokens - 2, (wav.length / RATE) * tokPerS + 8));
    let st = this.initState(1), lg;
    if (language === null) {
      [lg, st] = await this.step([[this.bos]], cross, st);
      language = this.languages.reduce((a, b) => (lg[0][this.tags[b]] > lg[0][this.tags[a]] ? b : a));
      [lg, st] = await this.step([[this.tags[language]]], cross, st);
    } else {
      [lg, st] = await this.step([[this.bos, this.tags[language]]], cross, st);
    }
    const mask = (rows) => rows.forEach((r) => r.fill(-Infinity, this.text));
    mask(lg);
    if (beams <= 1) {
      let ids = [];
      for (let i = 0; i < cap; i++) {
        let k = 0;
        for (let j = 1; j < this.text; j++) if (lg[0][j] > lg[0][k]) k = j;
        if (k === this.eos) break;
        ids.push(k);
        [lg, st] = await this.step([[k]], cross, st);
        mask(lg);
      }
      ids = tidy(ids);
      return { text: this.detok(ids), language, tokens: ids };
    }
    const norm = lengthNorm ? (sc, s) => sc / (s.length + 1) : (sc) => sc;
    const lp = Shruti.logSoftmax(lg[0]);
    const first = [...lp.keys()].sort((a, b) => lp[b] - lp[a] || a - b).slice(0, beams);
    let seqs = first.map((k) => [k]), scores = first.map((k) => lp[k]);
    const finished = [];
    seqs.forEach((s, i) => { if (s[s.length - 1] === this.eos) finished.push([scores[i], s.slice(0, -1)]); });
    const live = seqs.map((s, i) => i).filter((i) => seqs[i][seqs[i].length - 1] !== this.eos);
    seqs = live.map((i) => seqs[i]); scores = live.map((i) => scores[i]);
    st = this.pick(st, seqs.map(() => 0));
    for (let it = 0; it < cap - 1; it++) {
      if (!seqs.length) break;
      [lg, st] = await this.step(seqs.map((s) => [s[s.length - 1]]), cross, st);
      mask(lg);
      // the 2K best (beam, token) candidates, best first
      const top = [];
      lg.forEach((row, b) => {
        const l = Shruti.logSoftmax(row);
        for (let k = 0; k < l.length; k++) {
          const v = scores[b] + l[k];
          if (top.length < 2 * beams || v > top[top.length - 1][0]) {
            top.push([v, b, k]);
            top.sort((x, y) => y[0] - x[0] || x[1] * l.length + x[2] - (y[1] * l.length + y[2]));
            if (top.length > 2 * beams) top.pop();
          }
        }
      });
      const newSeqs = [], newScores = [], src = [];
      for (const [v, b, k] of top) {
        if (k === this.eos) finished.push([v, seqs[b]]);
        else { newSeqs.push([...seqs[b], k]); newScores.push(v); src.push(b); }
        if (newSeqs.length === beams) break;
      }
      const bestDone = finished.length ? Math.max(...finished.map(([sc, s]) => norm(sc, s))) : -Infinity;
      if (finished.length >= beams && (!newScores.length || Math.max(...newScores.map((sc, i) => norm(sc, newSeqs[i]))) < bestDone)) break;
      seqs = newSeqs; scores = newScores;
      st = this.pick(st, src);
    }
    const pool = finished.length ? finished : scores.map((sc, i) => [sc, seqs[i]]);
    let best = pool[0];
    for (const f of pool) if (norm(...f) > norm(...best)) best = f;
    const ids = tidy(best[1]);
    return { text: this.detok(ids), language, tokens: ids };
  }
}

// WAV (PCM 8/16/24/32-bit or float32) -> Float32Array at 16 kHz mono (linear resampling)
export function decodeWav(buf) {
  const dv = new DataView(buf);
  let p = 12, fmt = null, data = null;
  while (p + 8 <= dv.byteLength) {
    const id = String.fromCharCode(...new Uint8Array(buf, p, 4)), len = dv.getUint32(p + 4, true);
    if (id === "fmt ") fmt = { tag: dv.getUint16(p + 8, true), ch: dv.getUint16(p + 10, true), rate: dv.getUint32(p + 12, true), bits: dv.getUint16(p + 22, true) };
    if (id === "data") data = [p + 8, Math.min(len, dv.byteLength - p - 8)];
    p += 8 + len + (len % 2);
  }
  const bps = fmt.bits / 8, n = Math.floor(data[1] / bps / fmt.ch), x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < fmt.ch; c++) {
      const o = data[0] + (i * fmt.ch + c) * bps;
      if (fmt.tag === 3) s += dv.getFloat32(o, true);
      else if (bps === 1) s += (dv.getUint8(o) - 128) / 128;
      else if (bps === 2) s += dv.getInt16(o, true) / 32768;
      else if (bps === 3) s += ((dv.getUint8(o) | (dv.getUint8(o + 1) << 8) | (dv.getInt8(o + 2) << 16))) / 8388608;
      else s += dv.getInt32(o, true) / 2147483648;
    }
    x[i] = s / fmt.ch;
  }
  return resample(x, fmt.rate);
}

export function resample(x, rate) {
  if (rate === RATE) return x;
  const n = Math.round((x.length * RATE) / rate), y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = (i * rate) / RATE, j = Math.floor(t), f = t - j;
    y[i] = j + 1 < x.length ? x[j] * (1 - f) + x[j + 1] * f : x[x.length - 1];
  }
  return y;
}
