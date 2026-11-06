import { $, calm, canvas, onScheme, sleep, tone } from "./dom.js";

// Public SM counts.
const GPUS = [
  ["H100 SXM", 132],
  ["H100 PCIe", 114],
  ["A100", 108],
  ["B200", 148],
  ["L40S", 142],
  ["L4", 58],
  ["RTX 4090", 128],
  ["RTX 5090", 170],
];
const state = { wave: -1, playing: false };

/** Tiles, waves and efficiency for an M x N output split into tm x tn tiles on `sms` SMs. */
export function plan(m, n, tm, tn, sms) {
  const tiles = Math.ceil(m / tm) * Math.ceil(n / tn);
  const waves = Math.ceil(tiles / sms);
  return { m, n, tm, tn, sms, tiles, waves, last: tiles - (waves - 1) * sms, efficiency: tiles / (waves * sms) };
}

function current() {
  const [tm, tn] = $("w-tile").value.split("x").map(Number);
  return plan(Number($("w-m").value), Number($("w-n").value), tm, tn, GPUS[Number($("w-gpu").value)][1]);
}

function drawSms(p) {
  const { ctx, width, height } = canvas($("w-sms"));
  const gridHeight = height - 40;
  const gap = 5;
  let cols = Math.ceil(Math.sqrt(p.sms * (width / gridHeight)));
  while (cols > 2 && Math.ceil(p.sms / (cols - 1)) * (width / (cols - 1)) <= gridHeight) cols--;
  const rows = Math.ceil(p.sms / cols);
  const side = Math.min((width - 24 - (cols - 1) * gap) / cols, (gridHeight - 12 - (rows - 1) * gap) / rows);
  const left = (width - cols * side - (cols - 1) * gap) / 2;
  const [ink, sheet, lost, sans] = [tone("ink"), tone("sheet"), tone("lost"), tone("sans")];
  const wave = state.wave < 0 ? p.waves - 1 : state.wave;
  const busy = Math.min(p.sms, p.tiles - wave * p.sms);
  for (let i = 0; i < p.sms; i++) {
    const x = left + (i % cols) * (side + gap);
    const y = 12 + Math.floor(i / cols) * (side + gap);
    if (i >= busy) {
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = lost;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x + 0.75, y + 0.75, side - 1.5, side - 1.5);
      ctx.setLineDash([]);
    } else {
      ctx.fillStyle = ink;
      ctx.fillRect(x, y, side, side);
    }
    if (i < busy && side > 30) {
      ctx.fillStyle = sheet;
      ctx.font = `700 ${Math.min(13, side / 3.4)}px ${sans}`;
      ctx.textAlign = "center";
      ctx.fillText(`t${wave * p.sms + i}`, x + side / 2, y + side / 2 + 4);
    }
  }
  ctx.fillStyle = busy < p.sms ? lost : ink;
  ctx.font = `700 15px ${sans}`;
  ctx.textAlign = "left";
  const idle = busy < p.sms ? `, ${p.sms - busy} idle` : "";
  ctx.fillText(`wave ${wave + 1} of ${p.waves}: ${busy} of ${p.sms} SMs busy${idle}`, left, height - 12);
}

function drawSawtooth(p) {
  const { ctx, width, height } = canvas($("w-saw"));
  const pad = { l: 52, r: 16, t: 14, b: 30 };
  const [step, top] = [128, 32768];
  const X = (n) => pad.l + ((n - step) / (top - step)) * (width - pad.l - pad.r);
  const Y = (e) => pad.t + (1 - e) * (height - pad.t - pad.b);
  const at = (n) => plan(p.m, n, p.tm, p.tn, p.sms).efficiency;
  const [ink, soft, faint, sheet] = [tone("ink"), tone("soft"), tone("faint"), tone("sheet")];
  ctx.font = `13px ${tone("sans")}`;
  ctx.strokeStyle = faint;
  ctx.fillStyle = soft;
  ctx.lineWidth = 1;
  for (const e of [0.25, 0.5, 0.75, 1]) {
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(e));
    ctx.lineTo(width - pad.r, Y(e));
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(`${e * 100}%`, pad.l - 8, Y(e) + 4);
  }
  const ticks = width < 560 ? [8192, 16384, top] : [4096, 8192, 16384, 24576, top];
  for (const n of ticks) {
    ctx.textAlign = n === top ? "right" : "center";
    const label = n === top && width >= 560 ? `N = ${n.toLocaleString()}` : n.toLocaleString();
    ctx.fillText(label, X(n), height - 8);
  }
  ctx.strokeStyle = ink;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let n = step; n <= top; n += step) ctx[n === step ? "moveTo" : "lineTo"](X(n), Y(at(n)));
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(X(p.n), Y(at(p.n)), 6, 0, 2 * Math.PI);
  ctx.fillStyle = ink;
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = sheet;
  ctx.stroke();
}

function update() {
  const p = current();
  $("w-m-v").textContent = p.m.toLocaleString();
  $("w-n-v").textContent = p.n.toLocaleString();
  $("w-tiles").textContent = p.tiles.toLocaleString();
  $("w-waves").textContent = (p.tiles / p.sms).toFixed(2);
  $("w-last").textContent = `${p.last} of ${p.sms}`;
  $("w-eff").textContent = `${(100 * p.efficiency).toFixed(1)}%`;
  drawSms(p);
  drawSawtooth(p);
}

async function play() {
  if (state.playing) return;
  state.playing = true;
  $("w-play").disabled = true;
  const p = current();
  for (let w = 0; w < p.waves && state.playing; w++) {
    state.wave = w;
    drawSms(p);
    await sleep(calm ? 200 : Math.max(250, 2400 / p.waves));
  }
  state.playing = false;
  state.wave = -1;
  $("w-play").disabled = false;
  drawSms(current());
}

export function initWaves() {
  for (const [i, [name, sms]] of GPUS.entries()) $("w-gpu").add(new Option(`${name} (${sms} SMs)`, i));
  for (const id of ["w-gpu", "w-tile", "w-m", "w-n"]) {
    $(id).addEventListener("input", () => {
      state.playing = false;
      state.wave = -1;
      update();
    });
  }
  $("w-play").addEventListener("click", play);
  new ResizeObserver(update).observe($("w-sms"));
  onScheme(update);
  // Canvas text needs the web fonts, so draw again once they land.
  document.fonts.ready.then(update);
}
