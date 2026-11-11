import { $, canvas, json, onScheme, tone } from "./dom.js";

const state = { trace: null, run: 0 };
const FORMATS = {
  int8: { bits: 8 },
  "fp8 e4m3": { bits: 8, fp8: true },
  int4: { bits: 4 },
  int3: { bits: 3 },
  int2: { bits: 2 },
};

// Every non-negative fp8 e4m3 value, subnormals first.
const E4M3 = [0];
for (let m = 1; m < 8; m++) E4M3.push((m / 8) * 2 ** -6);
for (let e = -6; e <= 8; e++) {
  for (let m = 0; m < 8; m++) if ((1 + m / 8) * 2 ** e <= 448) E4M3.push((1 + m / 8) * 2 ** e);
}

/** Rounds to the nearest whole number with ties going to the even one, as torch.round does. */
function even(x) {
  const below = Math.floor(x);
  const rest = x - below;
  return rest > 0.5 || (rest === 0.5 && below % 2 !== 0) ? below + 1 : below;
}

/** Rounds to the nearest fp8 e4m3 value: 3 mantissa bits, smallest normal 2^-6, largest 448. */
function e4m3(x) {
  if (x === 0) return 0;
  const step = 2 ** (Math.max(Math.floor(Math.log2(Math.abs(x))), -6) - 3);
  return Math.sign(x) * Math.min(448, even(Math.abs(x) / step) * step);
}

/**
 * Rounds each group of `values` the way switchboard/quant.py does. Integers spread 2^bits levels from the group's min
 * to its max, and fp8 scales the group so its biggest weight becomes 448.
 */
export function quantize(values, { bits, fp8 }, group) {
  const out = new Float64Array(values.length);
  const grids = [];
  for (let s = 0; s < values.length; s += group) {
    const g = values.slice(s, s + group);
    let levels;
    if (fp8) {
      const scale = Math.max(...g.map(Math.abs)) / 448 || 1e-12;
      g.forEach((v, i) => {
        out[s + i] = e4m3(v / scale) * scale;
      });
      levels = { scale };
    } else {
      const lo = Math.min(...g);
      const step = (Math.max(...g) - lo) / (2 ** bits - 1) || 1e-12;
      g.forEach((v, i) => {
        out[s + i] = even((v - lo) / step) * step + lo;
      });
      levels = { lo, step, count: 2 ** bits };
    }
    grids.push({ start: s, end: s + g.length, ...levels });
  }
  return { out, grids };
}

/** The grid values of one group that fall between `min` and `max`. */
function gridValues(grid, min, max) {
  if (grid.count) {
    const from = Math.max(0, Math.ceil((min - grid.lo) / grid.step));
    const to = Math.min(grid.count - 1, Math.floor((max - grid.lo) / grid.step));
    return Array.from({ length: Math.max(0, to - from + 1) }, (_, k) => grid.lo + (from + k) * grid.step);
  }
  return E4M3.flatMap((v) => (v ? [v, -v] : [0]))
    .map((v) => v * grid.scale)
    .filter((v) => v >= min && v <= max);
}

function settings() {
  const { values: row, where } = state.trace.rows[Number($("q-row").value)];
  const format = FORMATS[$("q-format").value];
  const group = $("q-group").value === "row" ? row.length : Number($("q-group").value);
  return { row, where, format, group };
}

function draw() {
  const { row, where, format, group } = settings();
  const { out, grids } = quantize(row, format, group);
  let peak = 0;
  row.forEach((v, i) => {
    if (Math.abs(v) > Math.abs(row[peak])) peak = i;
  });
  const start = Math.floor(peak / 128) * 128;
  const shown = row.slice(start, start + 128);

  const { ctx, width, height } = canvas($("q-plot"));
  const pad = { l: 64, r: 12, t: 14, b: 34 };
  const all = [...shown, ...out.subarray(start, start + 128)];
  const [lo, hi] = [Math.min(...all), Math.max(...all)];
  const span = hi - lo || 1;
  const [min, max] = [lo - span * 0.06, hi + span * 0.06];
  const X = (i) => pad.l + ((i + 0.5) / 128) * (width - pad.l - pad.r);
  const Y = (v) => pad.t + ((max - v) / (max - min)) * (height - pad.t - pad.b);
  const [ink, soft, faint, lost, sans] = [tone("ink"), tone("soft"), tone("faint"), tone("lost"), tone("sans")];

  // 408: the grid each group rounds to, where its lines are far enough apart to see.
  ctx.strokeStyle = faint;
  ctx.lineWidth = 1;
  let [drawn, hidden] = [0, 0];
  for (const grid of grids) {
    if (grid.end <= start || grid.start >= start + 128) continue;
    const values = gridValues(grid, min, max).toSorted((a, b) => a - b);
    const gap = Math.min(...values.slice(1).map((v, k) => v - values[k]));
    if (gap * ((height - pad.t - pad.b) / (max - min)) < 3) {
      hidden++;
      continue;
    }
    drawn++;
    const [a, b] = [X(Math.max(grid.start, start) - start) - 2, X(Math.min(grid.end, start + 128) - 1 - start) + 2];
    ctx.beginPath();
    for (const v of values) {
      ctx.moveTo(a, Y(v));
      ctx.lineTo(b, Y(v));
    }
    ctx.stroke();
  }

  ctx.font = `13px ${sans}`;
  ctx.fillStyle = soft;
  ctx.textAlign = "right";
  // Zero gets a label only when it sits clear of the two ends.
  for (const v of [hi, lo, 0]) {
    if (v === 0 && Math.min(Math.abs(Y(0) - Y(hi)), Math.abs(Y(0) - Y(lo))) < 18) continue;
    ctx.fillText(v.toFixed(3), pad.l - 10, Y(v) + 4);
  }
  ctx.textAlign = "left";
  const note = hidden ? `, ${hidden} of ${drawn + hidden} grids too fine to draw` : "";
  const full = `weights ${start} to ${start + 127} of ${row.length} in ${where}${note}`;
  const short = `weights ${start} to ${start + 127} of ${row.length}`;
  ctx.fillText(ctx.measureText(full).width < width - pad.l ? full : short, pad.l, height - 10);

  shown.forEach((v, i) => {
    const q = out[start + i];
    // 406: what rounding lost.
    ctx.strokeStyle = lost;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(X(i), Y(v));
    ctx.lineTo(X(i), Y(q));
    ctx.stroke();
    // 404: the rounded weight.
    ctx.fillStyle = ink;
    ctx.fillRect(X(i) - 3.5, Y(q) - 1, 7, 2);
    // 402: the weight as stored in bf16.
    ctx.beginPath();
    ctx.arc(X(i), Y(v), 2.6, 0, 2 * Math.PI);
    ctx.fill();
  });

  const home = grids.find((g) => g.start <= peak && peak < g.end);
  const used = new Set(Array.from(out.subarray(home.start, home.end), (v) => v.toPrecision(6))).size;
  const scales = format.fp8 ? 16 : 32;
  const rms = (list) => Math.sqrt(list.reduce((sum, v) => sum + v * v, 0) / list.length);
  $("q-bits").textContent = (format.bits + scales / group).toFixed(2);
  $("q-used").textContent = format.fp8 ? used : `${used} of ${2 ** format.bits}`;
  $("q-err").textContent = `${(100 * (rms(row.map((v, i) => v - out[i])) / rms(row))).toFixed(1)}%`;
}

function pickRun(index) {
  state.run = index;
  const run = state.trace.runs[index];
  for (const [i, button] of document.querySelectorAll("#q-runs button").entries()) {
    button.setAttribute("aria-pressed", i === index);
  }
  $("q-text").replaceChildren(
    ...run.tokens.map((text, i) =>
      Object.assign(document.createElement("span"), {
        textContent: text,
        className: i < run.same ? "" : i === run.same ? "miss" : "after",
      }),
    ),
  );
  $("q-say").textContent =
    run.bits === 16
      ? "Qwen3-0.6B as it ships. Every other format is compared with this text."
      : run.same === run.tokens.length
        ? `Writes exactly what bf16 writes, all ${run.tokens.length} tokens of it.`
        : run.same === 0
          ? "Picks a different first token from bf16 and goes its own way from there."
          : `Writes what bf16 writes for ${run.same} tokens, then picks a different one and goes its own way.`;
}

export async function initQuant() {
  state.trace = await json("data/quant.json");
  const { rows, runs } = state.trace;
  for (const i of rows.keys()) {
    $("q-row").add(new Option(i ? "A typical row" : "The row with the biggest outlier", i));
  }
  const body = $("q-runs");
  for (const [i, run] of runs.entries()) {
    const tr = body.insertRow();
    const button = Object.assign(document.createElement("button"), { textContent: run.format, className: "chip" });
    button.addEventListener("click", () => pickRun(i));
    tr.insertCell().append(button);
    tr.insertCell().textContent =
      run.bits === 16 ? "16" : (run.bits + (run.fp8 ? 16 : 32) / (run.group || 1024)).toFixed(2);
    tr.insertCell().textContent = `${(100 * run.agree).toFixed(1)}%`;
    tr.insertCell().textContent = run.kl.toFixed(4);
    tr.insertCell().textContent = run.same;
  }
  for (const id of ["q-row", "q-format", "q-group"]) $(id).addEventListener("input", draw);
  new ResizeObserver(draw).observe($("q-plot"));
  onScheme(draw);
  document.fonts.ready.then(draw);
  pickRun(Math.min(3, runs.length - 1));
}
