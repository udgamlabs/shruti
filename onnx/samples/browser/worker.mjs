// Runs the model off the main thread, so the page stays responsive while it decodes.
const CDN = "https://cdn.jsdelivr.net/npm";
const ORT = `${CDN}/onnxruntime-web@1.30.0/dist/`;  // pinned
const local = new URLSearchParams(location.search).has("local");  // ?local reads the package and models from a source checkout, for testing
const PACKAGE = local ? "../../js/index.mjs" : `${CDN}/shruti-onnx@0.2.0/index.mjs`;
const PACKAGE_FROM_GITHUB = "https://cdn.jsdelivr.net/gh/udgamlabs/shruti@main/onnx/js/index.mjs";  // the same file, if the npm copy can't be reached

const libraries = Promise.all([import(`${ORT}ort.wasm.min.mjs`), import(PACKAGE).catch(() => import(PACKAGE_FROM_GITHUB))]).then(([ort, shruti]) => {
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;  // threads need cross-origin isolation headers, which most hosts don't send
  return { ort, shruti };
});
let asr = null;

async function checkedFetch(url) {
  const name = url.split("/").pop();
  postMessage({ type: "status", text: `Loading ${name}…` });
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}. Is this language available yet?`);
  return r;
}

onmessage = async ({ data }) => {
  try {
    const { ort, shruti } = await libraries;
    if (data.type === "languages") postMessage({ type: "languages", languages: shruti.LANGUAGES });
    if (data.type === "load") {
      const t0 = performance.now();
      const options = { ort, language: data.language, variant: data.variant, fetchFn: checkedFetch };
      if (local) options.baseUrl = `../../hf/${data.language}/${data.variant === "int8" ? "onnx-int8" : "onnx"}/`;
      asr = await shruti.createShruti(options);
      postMessage({ type: "loaded", ms: performance.now() - t0 });
    }
    if (data.type === "transcribe") {
      const t0 = performance.now();
      const out = await asr.transcribe(data.samples, { language: data.language });
      postMessage({ type: "result", text: out.text, language: out.language, ms: performance.now() - t0, seconds: data.samples.length / 16000 });
    }
  } catch (e) {
    postMessage({ type: "error", text: e.message, canRetry: data.type === "transcribe" });
  }
};
