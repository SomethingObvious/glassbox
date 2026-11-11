import { $, json } from "./dom.js";

const state = { trace: null, run: 0, at: [0, 0] };

/** The first layer from which the last layer's top guess stays on top all the way down. */
function settles(layers, position) {
  const final = layers.at(-1)[position][0][0];
  let from = layers.length - 1;
  while (from > 0 && layers[from - 1][position][0][0] === final) from--;
  return from;
}

function shown(text) {
  return text.replaceAll(" ", "·").replaceAll("\n", "\\n") || "·";
}

function pick(layer, position) {
  const run = state.trace.runs[state.run];
  const top = run.layers[layer][position];
  state.at = [layer, position];
  for (const cell of $("l-grid").querySelectorAll("[aria-current]")) cell.removeAttribute("aria-current");
  $("l-grid").querySelector(`[data-at="${layer}-${position}"]`).setAttribute("aria-current", "true");
  $("l-where").replaceChildren(
    `After layer ${layer + 1} of ${run.layers.length}, at `,
    Object.assign(document.createElement("b"), { textContent: shown(run.tokens[position]) }),
  );
  $("l-top").replaceChildren(
    ...top.map(([text, p]) => {
      const li = document.createElement("li");
      const bar = Object.assign(document.createElement("i"), { className: "l-bar" });
      bar.style.width = `${100 * p}%`;
      li.append(
        Object.assign(document.createElement("span"), { className: "l-tok", textContent: shown(text) }),
        Object.assign(document.createElement("span"), { textContent: `${(100 * p).toFixed(1)}%` }),
        bar,
      );
      return li;
    }),
  );
}

function pickRun(index) {
  state.run = index;
  const run = state.trace.runs[index];
  for (const [i, chip] of [...$("l-runs").children].entries()) chip.setAttribute("aria-pressed", i === index);
  const grid = $("l-grid");
  grid.style.setProperty("--cols", run.tokens.length);
  const cells = [document.createElement("span")];
  for (const text of run.tokens) {
    cells.push(Object.assign(document.createElement("span"), { className: "l-head", textContent: shown(text) }));
  }
  run.layers.forEach((layer, l) => {
    const label = l % 4 === 3 || l === 0 || l === run.layers.length - 1 ? String(l + 1) : "";
    cells.push(Object.assign(document.createElement("span"), { className: "l-layer", textContent: label }));
    layer.forEach(([[text, p]], position) => {
      const cell = Object.assign(document.createElement("span"), { className: "l-cell", textContent: shown(text) });
      cell.dataset.at = `${l}-${position}`;
      // The shade jumps at one half so the text on it can switch colour and stay readable.
      const shade = p < 0.5 ? 8 + 50 * p : 75 + 25 * p;
      cell.style.background = `color-mix(in srgb, var(--ink) ${shade.toFixed(0)}%, transparent)`;
      if (p >= 0.5) cell.classList.add("on");
      for (const kind of ["pointerenter", "click"]) cell.addEventListener(kind, () => pick(l, position));
      cells.push(cell);
    });
  });
  grid.replaceChildren(...cells);
  // On a narrow screen the grid scrolls, and the answer is in the last column.
  grid.parentElement.scrollLeft = grid.scrollWidth;
  const last = run.tokens.length - 1;
  const from = settles(run.layers, last);
  const answer = run.layers.at(-1)[last][0][0].trim();
  $("l-settle").textContent =
    from === run.layers.length - 1
      ? `The answer "${answer}" only reaches the top in the last layer.`
      : `The answer "${answer}" reaches the top at layer ${from + 1} and stays there through layer ${run.layers.length}.`;
  pick(run.layers.length - 1, last);
}

export async function initLens() {
  state.trace = await json("data/lens.json");
  $("l-runs").replaceChildren(
    ...state.trace.runs.map((run, i) => {
      const chip = Object.assign(document.createElement("button"), {
        className: "chip",
        textContent: run.text.length > 34 ? `${run.text.slice(0, 32)}...` : run.text,
        title: run.text,
      });
      chip.addEventListener("click", () => pickRun(i));
      return chip;
    }),
  );
  pickRun(0);
  $("l-grid").addEventListener("keydown", (event) => {
    const step = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[event.key];
    if (!step) return;
    event.preventDefault();
    const run = state.trace.runs[state.run];
    const layer = Math.max(0, Math.min(run.layers.length - 1, state.at[0] + step[0]));
    const position = Math.max(0, Math.min(run.tokens.length - 1, state.at[1] + step[1]));
    pick(layer, position);
    $("l-grid").querySelector("[aria-current]").scrollIntoView({ block: "nearest", inline: "nearest" });
  });
}
