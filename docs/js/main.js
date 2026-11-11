import { $ } from "./dom.js";
import { initDraft } from "./draft.js";
import { initLens } from "./lens.js";
import { initQuant } from "./quant.js";
import { initRoute } from "./route.js";
import { initWaves } from "./waves.js";

// Pages sends the deploy time as Last-Modified, so the running head dates itself.
const issued = new Date(document.lastModified);
if (!Number.isNaN(issued.getTime())) {
  $("issued").textContent = issued.toLocaleDateString("en-CA", { month: "long", year: "numeric" });
  $("issued").dateTime = issued.toISOString().slice(0, 7);
}

initWaves();
await Promise.all([initDraft(), initRoute(), initQuant(), initLens()]);
window.__done = true;
