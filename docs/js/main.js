import { $ } from "./dom.js";
import { initDraft } from "./draft.js";
import { initLens } from "./lens.js";
import { initQuant } from "./quant.js";
import { initRoute } from "./route.js";
import { initWaves } from "./waves.js";

// Pages sends the deploy time as Last-Modified, so the running head dates itself, with the day it was in Toronto.
const issued = new Date(document.lastModified);
if (!Number.isNaN(issued.getTime())) {
  const toronto = { timeZone: "America/Toronto" };
  $("issued").textContent = issued.toLocaleDateString("en-CA", { ...toronto, dateStyle: "long" });
  // Canadian English writes a plain date year first with hyphens, which is the form the attribute takes.
  $("issued").dateTime = issued.toLocaleDateString("en-CA", toronto);
}

initWaves();
await Promise.all([initDraft(), initRoute(), initQuant(), initLens()]);
window.__done = true;
