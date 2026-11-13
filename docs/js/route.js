import { $, calm, json, svg } from "./dom.js";
import { LIBRARY, pieces, readable } from "./tokens.js";
import { cached, runtime } from "./weights.js";

const state = { trace: null, run: 0, current: 0, playing: true, whole: false, seen: false, cells: [], shown: [] };
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
        const cell = svg("g", { style: `--l: ${l}` }, grid);
        const box = { x, y, width: cw - 3, height: ch - 3 };
        svg("rect", { ...box, class: "g-cell" }, cell);
        // These are styles and not attributes, so that the stylesheet's transitions run when they change.
        const heat = svg("rect", { ...box, class: "g-heat", style: "fill-opacity: 0.03" }, cell);
        const pick = svg("rect", { ...box, class: "g-pick", style: "opacity: 0" }, cell);
        const ring = svg(
          "circle",
          { cx: x + (cw - 3) / 2, cy: y + (ch - 3) / 2, r: 3, class: "g-ring", style: "opacity: 0" },
          cell,
        );
        cell.addEventListener("pointerenter", (event) => tooltip(event, l, e));
        cell.addEventListener("pointerleave", () => {
          $("tip").style.opacity = 0;
        });
        return { cell, heat, pick, ring };
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
      : `layer ${l}, expert ${e}: weight ${layer.weights[state.current][slot].toFixed(3)} for "${state.shown[state.current].text}"`;
  Object.assign(tip.style, { left: `${event.clientX + 14}px`, top: `${event.clientY + 14}px`, opacity: 1 });
}

/** Moves the keyboard's cell in the grid and shows its tooltip beside it. */
function focusCell(l, e) {
  const clamp = (v, n) => Math.max(0, Math.min(n - 1, v));
  state.focus = [clamp(l, state.cells.length), clamp(e, state.cells[0].length)];
  state.focused?.classList.remove("g-focus");
  state.focused = state.cells[state.focus[0]][state.focus[1]].cell;
  state.focused.classList.add("g-focus");
  const box = state.focused.getBoundingClientRect();
  tooltip({ clientX: box.right, clientY: box.bottom }, ...state.focus);
}

function pause() {
  state.playing = false;
  $("r-play").textContent = "Play";
}

function setWhole(whole) {
  state.whole = whole;
  $("r-mode").setAttribute("aria-pressed", whole);
  $("r-mode").textContent = whole ? "Show One Token" : "Show Whole Sentence";
  show(state.current);
}

function pickRun(index) {
  state.run = index;
  const run = state.trace.runs[index];
  for (const [i, chip] of [...$("r-runs").children].entries()) chip.setAttribute("aria-pressed", i === index);
  state.shown = readable(run.tokens);
  $("r-toks").replaceChildren(
    ...state.shown.map(({ text, note }, j) => {
      const button = Object.assign(document.createElement("button"), {
        className: note ? "tok partial" : "tok",
        textContent: text.replaceAll(" ", "·") || "·",
        title: note,
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
  // The faint wash is the sentence's total router weight on each expert.
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
      state.cells[l][e].heat.style.fillOpacity = (0.03 + (0.3 * value) / peak).toFixed(3);
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
      cell.pick.style.opacity = lit ? (0.45 + 0.55 * Math.min(1, weight * 3)).toFixed(2) : 0;
      cell.ring.style.opacity = lit && before.has(e) ? 1 : 0;
      if (lit && before.has(e)) shared++;
    });
  });
  const { text, note } = state.shown[j];
  $("r-cur").textContent = text.trim() || "space";
  $("r-cur").classList.toggle("partial", Boolean(note));
  $("r-share").textContent = note
    ? note
    : j === 0
      ? "The first token has no token before it to compare with."
      : `It shares ${(shared / run.layers.length).toFixed(1)} of its ${state.trace.k} experts per layer with the token before it.`;
}

/** The Hacker News headline that a daily workflow routed and pushed to the live branch. */
async function headline() {
  try {
    const live = await json(LIVE);
    const { day, link, place = 1 } = live.headline;
    const note = $("r-live");
    note.replaceChildren(
      `The first sentence is today's ${place > 1 ? `number ${place}` : "top"} `,
      Object.assign(document.createElement("a"), { href: link, textContent: "Hacker News headline" }),
      `, routed on ${day} by a daily workflow that runs the same model.`,
    );
    note.hidden = false;
    return { ...live.runs[0], live: true };
  } catch {
    // The saved sentences are enough on their own.
    return null;
  }
}

const LIVE = "https://raw.githubusercontent.com/SomethingObvious/switchboard/live/route.json";
const GRANITE = "ibm-granite/granite-3.0-1b-a400m-instruct";
// Granite with its experts in 8 bits and its routers in fp32, made by `python -m switchboard routes`.
const ROUTES = "https://huggingface.co/hoyporti/granite-switchboard/resolve/4e76cb44b756143a3c806ec4555507791006acfa";
const live = { ort: null, tok: null, session: null, gpu: false };

async function loadRouter() {
  const say = (text) => {
    $("r-status").textContent = text;
  };
  say("Loading the tokenizer and runtime...");
  live.ort ??= await runtime();
  live.tok ??= await (await import(LIBRARY)).AutoTokenizer.from_pretrained(GRANITE);
  if (live.session) return;
  const gb = (n) => (n / 1e9).toFixed(2);
  const [graph, data] = await Promise.all([
    cached(`${ROUTES}/routes.onnx`, () => {}),
    cached(`${ROUTES}/routes.onnx.data`, (loaded, total) =>
      say(`Downloading 8-bit weights, ${gb(loaded)} of ${gb(total)} GB`),
    ),
  ]);
  say("Loading the model onto the GPU...");
  live.gpu = Boolean(await navigator.gpu?.requestAdapter());
  live.session = await live.ort.InferenceSession.create(graph, {
    executionProviders: live.gpu ? ["webgpu"] : ["wasm"],
    logSeverityLevel: 3,
    externalData: [{ path: "routes.onnx.data", data }],
  });
}

/** Routes the visitor's sentence and adds it as the first chip, laid out like a saved trace. */
async function routeYours(event) {
  event.preventDefault();
  const text = $("r-text").value.trim();
  if (!text) return;
  $("r-go").disabled = true;
  try {
    await loadRouter();
    const ids = live.tok.encode(text, { add_special_tokens: false });
    const started = performance.now();
    const input = new live.ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
    const { router_logits: logits } = await live.session.run({ input_ids: input });
    const [layers, n, experts] = logits.dims;
    const k = state.trace.k;
    const run = { text, tokens: pieces(live.tok, ids), layers: [], yours: true };
    for (let l = 0; l < layers; l++) {
      const layer = { experts: [], weights: [] };
      for (let p = 0; p < n; p++) {
        const row = logits.data.subarray((l * n + p) * experts, (l * n + p + 1) * experts);
        const top = [...row.keys()].sort((a, b) => row[b] - row[a]).slice(0, k);
        const exps = top.map((e) => Math.exp(row[e] - row[top[0]]));
        const sum = exps.reduce((a, b) => a + b, 0);
        layer.experts.push(top);
        layer.weights.push(exps.map((v) => v / sum));
      }
      run.layers.push(layer);
    }
    const ms = performance.now() - started;
    state.trace.runs = [run, ...state.trace.runs.filter((r) => !r.yours)];
    chips();
    pickRun(0);
    $("r-status").textContent = `Routed ${ids.length} tokens through 24 layers in ${ms.toFixed(0)} ms on this ${
      live.gpu ? "GPU" : "CPU"
    }. Try another.`;
  } catch (error) {
    $("r-status").textContent = `Couldn't run the router (${error.message}).`;
  } finally {
    $("r-go").disabled = false;
  }
}

function chips() {
  $("r-runs").replaceChildren(
    ...state.trace.runs.map((run, i) => {
      const label = run.live ? `Today on HN: ${run.text}` : run.yours ? `Yours: ${run.text}` : run.text;
      const chip = Object.assign(document.createElement("button"), {
        className: "chip",
        textContent: label.length > 34 ? `${label.slice(0, 32)}...` : label,
        title: run.text,
      });
      chip.addEventListener("click", () => pickRun(i));
      return chip;
    }),
  );
}

export async function initRoute() {
  const [saved, today] = await Promise.all([json("data/route.json"), headline()]);
  state.trace = saved;
  if (today) saved.runs.unshift(today);
  const { runs, experts } = state.trace;
  chips();
  $("r-form").addEventListener("submit", routeYours);
  drawGrid(experts, runs[0].layers.length);
  const grid = $("r-grid");
  // A mouse click also focuses the grid, and only keyboard focus should bring up a tooltip.
  grid.addEventListener("focus", () => grid.matches(":focus-visible") && focusCell(...(state.focus ?? [0, 0])));
  grid.addEventListener("blur", () => {
    $("tip").style.opacity = 0;
    state.focused?.classList.remove("g-focus");
  });
  grid.addEventListener("keydown", (event) => {
    const step = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[event.key];
    if (!step) return;
    event.preventDefault();
    const [l, e] = state.focus ?? [0, 0];
    focusCell(l + step[0], e + step[1]);
  });
  $("r-play").addEventListener("click", () => {
    state.playing = !state.playing;
    $("r-play").textContent = state.playing ? "Pause" : "Play";
    // Playing steps through the tokens, which the whole sentence view has none of.
    if (state.playing && state.whole) setWhole(false);
  });
  $("r-mode").addEventListener("click", () => {
    setWhole(!state.whole);
    if (state.whole) pause();
  });
  pickRun(0);
  // The grid only steps while it is on screen. Off screen it would still redraw, and that takes time from a run of
  // Figure 1 on the same GPU.
  new IntersectionObserver(([entry]) => {
    state.seen = entry.isIntersecting;
  }).observe(grid);
  setInterval(
    () => {
      if (state.seen && state.playing && !state.whole) {
        show((state.current + 1) % state.trace.runs[state.run].tokens.length);
      }
    },
    calm ? 2400 : 900,
  );
}
