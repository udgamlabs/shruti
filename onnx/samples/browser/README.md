# Shruti in the browser

One page with a language picker. No build step. After the first load, every language you have loaded works offline.

```bash
python3 -m http.server 8000    # then open http://localhost:8000
```

Host it anywhere that serves `https`. The microphone and offline support need `https` or `localhost`.

| File | What it does |
|---|---|
| `index.html` | The page: language picker, file button, microphone button. Any audio format your browser can decode |
| `worker.mjs` | Runs the model off the main thread, so the page stays responsive |
| `sw.js` | Offline support: keeps the page, the runtime and each model you have loaded |

**Downloads, once each:** the runtime (14 MB) and each language you load (32 MB, or 53 MB for the int8 build).
