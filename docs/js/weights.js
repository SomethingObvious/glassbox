const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";

/**
 * onnxruntime-web with its wasm swapped for the build in docs/ort, which is 1.31.1 plus rows.patch. Stock 1.31.1
 * reads every 4-bit weight again for each extra row of a pass, and the patch reads them once for up to 4 rows.
 */
export async function runtime() {
  const ort = await import(ORT);
  ort.env.logLevel = "error";
  // A bare path would be looked up beside the script on the CDN.
  ort.env.wasm.wasmPaths = new URL("../ort/", import.meta.url).href;
  return ort;
}

/** Reads a response into one buffer. A single 1.5 GB Blob can fail to read back in Chrome, so this avoids Blobs. */
async function read(response, progress) {
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    progress(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const part of parts.splice(0)) {
    bytes.set(part, at);
    at += part.length;
  }
  return bytes;
}

/** Fetches a file through the Cache API so the weights download once. The cache is a bonus and never an error. */
export async function cached(url, progress) {
  // The name is from when only Figure 2 kept weights, and a new one would strand what visitors already have.
  const cache = await caches.open("switchboard-routes").catch(() => null);
  const hit = await cache?.match(url).catch(() => null);
  if (hit) {
    try {
      return await read(hit, () => {});
    } catch {
      await cache.delete(url);
    }
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} answered ${response.status}`);
  const stored = cache?.put(url, response.clone()).catch(() => cache.delete(url));
  const bytes = await read(response, progress);
  await stored;
  return bytes;
}
