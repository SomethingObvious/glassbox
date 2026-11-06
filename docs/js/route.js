import { $, calm, json, svg } from "./dom.js";

const state = { trace: null, run: 0, current: 0, playing: true, whole: false, cells: [] };
const MONO = { class: "g-label" };

function drawGrid(experts, layers) {
  const grid = $("r-grid");
  const [cw, ch, left, top] = [30, 20, 60, 44];
  const width = left + experts * cw;
  grid.setAttribute("viewBox", `0 0 ${width} ${top + layers * ch}`);
  // Labels stay 12px on screen whatever width the grid is drawn at.
  new ResizeObserver(() => grid.style.setProperty("--label", `${(12 * width) / grid.clientWidth}px`)).observe(grid);
  for (let e = 0; e < experts; e += 8) {
    svg("text", { ...MONO, x: left + e * cw, y: top - 10 }, grid).textContent = e;
  }
  svg("text", { ...MONO, x: width - 3, y: top - 10, "text-anchor": "end" }, grid).textContent = experts - 1;
  for (let l = 0; l < layers; l++) {
    const y = top + l * ch;
    if (l % 4 === 0 || l === layers - 1) {
      svg("text", { ...MONO, x: left - 10, y: y + ch - 3, "text-anchor": "end" }, grid).textContent = l;
    }
    state.cells.push(
      Array.from({ length: experts }, (_, e) => {
        const x = left + e * cw;
        const cell = svg("g", {}, grid);
        const box = { x, y, width: cw - 3, height: ch - 3 };
        svg("rect", { ...box, class: "g-cell" }, cell);
        const heat = svg("rect", { ...box, class: "g-heat", "fill-opacity": 0.03 }, cell);
        const pick = svg("rect", { ...box, class: "g-pick", opacity: 0 }, cell);
        const ring = svg(
          "circle",
          { cx: x + (cw - 3) / 2, cy: y + (ch - 3) / 2, r: 3, class: "g-ring", opacity: 0 },
          cell,
        );
        cell.addEventListener("pointerenter", (event) => tooltip(event, l, e));
        cell.addEventListener("pointerleave", () => {
          $("tip").style.opacity = 0;
        });
        return { heat, pick, ring };
      }),
    );
  }
}

function tooltip(event, l, e) {
  const run = state.trace.runs[state.run];
  const layer = run.layers[l];
  const slot = layer.experts[state.current].indexOf(e);
  const users = layer.experts.filter((row) => row.includes(e)).length;
  const tip = $("tip");
  tip.textContent =
    state.whole || slot < 0
      ? `layer ${l}, expert ${e}: picked by ${users} of ${run.tokens.length} tokens`
      : `layer ${l}, expert ${e}: weight ${layer.weights[state.current][slot].toFixed(3)} for "${run.tokens[state.current]}"`;
  Object.assign(tip.style, { left: `${event.clientX + 14}px`, top: `${event.clientY + 14}px`, opacity: 1 });
}

function pause() {
  state.playing = false;
  $("r-play").textContent = "Play";
}

function pickRun(index) {
  state.run = index;
  const run = state.trace.runs[index];
  for (const [i, chip] of [...$("r-runs").children].entries()) chip.setAttribute("aria-pressed", i === index);
  $("r-toks").replaceChildren(
    ...run.tokens.map((text, j) => {
      const button = Object.assign(document.createElement("button"), {
        className: "tok",
        textContent: text.replaceAll(" ", "·") || "·",
      });
      for (const kind of ["pointerenter", "click"]) {
        button.addEventListener(kind, () => {
          pause();
          show(j);
        });
      }
      return button;
    }),
  );
  // Background heat: the sentence's total router weight on each expert.
  const use = run.layers.map((layer) => {
    const total = new Array(state.trace.experts).fill(0);
    layer.experts.forEach((row, t) => {
      row.forEach((e, k) => {
        total[e] += layer.weights[t][k];
      });
    });
    return total;
  });
  const peak = Math.max(...use.flat());
  use.forEach((row, l) => {
    row.forEach((value, e) => {
      state.cells[l][e].heat.setAttribute("fill-opacity", (0.03 + (0.3 * value) / peak).toFixed(3));
    });
  });
  const woken = run.layers.reduce((sum, layer) => sum + new Set(layer.experts.flat()).size, 0) / run.layers.length;
  $("r-woke").textContent =
    `Its ${run.tokens.length} tokens wake up ${woken.toFixed(1)} of the ${state.trace.experts} experts in a layer on ` +
    "average, so a batch of tokens touches most of the weights even though each token uses a quarter.";
  show(0);
}

function show(j) {
  state.current = j;
  const run = state.trace.runs[state.run];
  for (const [k, chip] of [...$("r-toks").children].entries()) chip.setAttribute("aria-current", k === j);
  let shared = 0;
  run.layers.forEach((layer, l) => {
    const now = new Map(layer.experts[j].map((e, k) => [e, layer.weights[j][k]]));
    const before = new Set(j > 0 ? layer.experts[j - 1] : []);
    state.cells[l].forEach((cell, e) => {
      const weight = now.get(e);
      const lit = !state.whole && weight !== undefined;
      cell.pick.setAttribute("opacity", lit ? (0.45 + 0.55 * Math.min(1, weight * 3)).toFixed(2) : 0);
      cell.ring.setAttribute("opacity", lit && before.has(e) ? 1 : 0);
      if (lit && before.has(e)) shared++;
    });
  });
  $("r-cur").textContent = run.tokens[j].trim() || "space";
  $("r-share").textContent =
    j === 0
      ? "The first token has no token before it to compare with."
      : `It shares ${(shared / run.layers.length).toFixed(1)} of its ${state.trace.k} experts per layer with the token before it.`;
}

export async function initRoute() {
  state.trace = await json("data/route.json");
  const { runs, experts } = state.trace;
  $("r-runs").replaceChildren(
    ...runs.map((run, i) => {
      const chip = Object.assign(document.createElement("button"), {
        className: "chip",
        textContent: run.text.length > 34 ? `${run.text.slice(0, 32)}...` : run.text,
      });
      chip.addEventListener("click", () => pickRun(i));
      return chip;
    }),
  );
  drawGrid(experts, runs[0].layers.length);
  $("r-play").addEventListener("click", () => {
    state.playing = !state.playing;
    $("r-play").textContent = state.playing ? "Pause" : "Play";
  });
  $("r-mode").addEventListener("click", () => {
    state.whole = !state.whole;
    $("r-mode").setAttribute("aria-pressed", state.whole);
    $("r-mode").textContent = state.whole ? "Show One Token" : "Show Whole Sentence";
    show(state.current);
  });
  pickRun(0);
  setInterval(
    () => {
      if (state.playing && !state.whole) show((state.current + 1) % runs[state.run].tokens.length);
    },
    calm ? 2400 : 900,
  );
}
