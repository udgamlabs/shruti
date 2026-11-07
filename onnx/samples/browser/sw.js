// Offline support. Nothing here sends your audio anywhere: it only keeps copies of files this page has already downloaded.
//   - The page itself: try the network, fall back to the saved copy, so a new version still reaches people who are online.
//   - The runtime and the model files: use the saved copy if there is one. They never change at a pinned version.
const CACHE = "shruti-sample-v1";
const SAVED_FOREVER = /(^|\.)(jsdelivr\.net|huggingface\.co|hf\.co)$/;

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

async function save(request, response) {
  if (response.ok) await (await caches.open(CACHE)).put(request, response.clone());
  return response;
}

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  const sameOrigin = url.origin === location.origin;
  const isModel = /\.onnx$|shruti\.json$/.test(url.pathname);
  if (sameOrigin && !isModel) {
    e.respondWith(fetch(url, { cache: "no-cache" }).then((r) => save(e.request, r)).catch(async () => {
      const copy = await caches.match(e.request, { ignoreSearch: false });
      if (copy) return copy;
      throw new Error("offline, and no saved copy");
    }));
  } else if (sameOrigin || SAVED_FOREVER.test(url.hostname)) {
    e.respondWith(caches.match(e.request).then((copy) => copy || fetch(e.request).then((r) => save(e.request, r))));
  }
});
