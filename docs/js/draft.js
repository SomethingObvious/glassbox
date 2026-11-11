import { $, calm, json, sleep } from "./dom.js";
import { LIBRARY, partial, pieces } from "./tokens.js";
import { cached, runtime } from "./weights.js";

// The same 4-bit export as onnx-community/Qwen3-1.7B-ONNX with its weights in a separate data file. The single
// 1.4 GB file runs out of WebAssembly memory while the session is created.
const TARGET = "Exempliss/Qwen3-1.7B-ONNX-Split";
const TARGET_REVISION = "f9b9656836a4315a690d0270f885b0758923bd13";
const WEIGHTS = `https://huggingface.co/${TARGET}/resolve/${TARGET_REVISION}/onnx/model_q4f16.onnx_data`;
// That graph alone and with the head after it, both made for graph capture by `python -m switchboard head`.
const HEAD =
  "https://huggingface.co/hoyporti/qwen3-1.7b-eagle3-switchboard/resolve/d9cbc9b3c9a73b1b71d746cfc15c64d6f9a620a3/onnx";
const STOP = new Set([151643, 151645]); // Qwen3 <|endoftext|> and <|im_end|>
// Checking more rows a pass costs WebGPU more than a second guess earns, as the second is right far less often.
const GUESSES = 1;
const ENTRIES = 1024; // cache rows on the GPU, the most a prompt and its answer can fill

const state = {
  ort: null,
  tok: null,
  big: null,
  pass: null,
  busy: false,
  replaying: false,
  raceId: 0,
};

const TODAY = "https://raw.githubusercontent.com/SomethingObvious/switchboard/live/draft.json";
const RACE = 20; // seconds a recorded run may take on screen before it plays faster than real time

/** How many guesses match the big model's greedy picks before the first miss. */
export function accepted(guesses, wants) {
  let kept = 0;
  while (kept < guesses.length && guesses[kept] === wants[kept]) kept++;
  return kept;
}

/** `message` is text, or a list of text and elements. */
function status(message, progress) {
  $("d-status").replaceChildren(...[message].flat());
  if (progress !== undefined) $("d-bar").style.transform = `scaleX(${progress})`;
}

function setBusy(busy) {
  state.busy = busy;
  $("d-run").disabled = $("d-replay").disabled = busy;
  for (const chip of $("d-tries").children) chip.disabled = busy;
}

function setReplaying(replaying) {
  state.replaying = replaying;
  // A replay that a live run cut short ends later, and it must not hand the button back mid-run.
  $("d-replay").disabled = replaying || state.busy;
}

const tokensIn = (steps) => steps.reduce((sum, s) => sum + s.kept + 1, 0);
const secondsIn = (steps) => steps.reduce((sum, s) => sum + s.seconds, 0);

function lane(name, tokens, total, seconds) {
  $(`race-${name}`).style.transform = `scaleX(${total ? tokens / total : 0})`;
  $(`race-${name}-n`).textContent = `${tokens} tokens, ${seconds.toFixed(2)} s`;
}

/** Lays out the words the run alone will write, each hidden until its token is done. */
function aloneText(words) {
  $("race-alone-t").replaceChildren(
    ...words.map((text) => Object.assign(document.createElement("span"), { textContent: partial(text)[0] })),
  );
}

/** Each pass as [the second it ends, tokens written by then]. */
function clock(seconds, tokens) {
  const marks = [];
  let [t, n] = [0, 0];
  for (const [i, s] of seconds.entries()) {
    t += s;
    n += tokens[i];
    marks.push([t, n]);
  }
  return marks;
}

/** Runs both lanes on the clock measured pass by pass, `speed` times faster than it ran. */
async function race({ steps, timing }, speed) {
  const id = ++state.raceId;
  const spec = clock(
    steps.map((s) => s.seconds),
    steps.map((s) => s.kept + 1),
  );
  const alone = clock(
    timing.alone,
    timing.alone.map(() => 1),
  );
  const total = spec.at(-1)[1];
  const words = steps.flatMap((s) => [...s.drafted.slice(0, s.kept), s.target]);
  // The trace only keeps the guessed run's tokens, which are the alone run's too when the texts match.
  aloneText(timing.same ? words : []);
  $("race-alone-box").hidden = !timing.same;
  const end = Math.max(spec.at(-1)[0], alone.at(-1)[0]);
  const started = performance.now();
  for (;;) {
    const now = calm ? end : ((performance.now() - started) / 1000) * speed;
    if (state.raceId !== id) return;
    for (const [name, marks] of [
      ["spec", spec],
      ["alone", alone],
    ]) {
      const tokens = marks.findLast(([t]) => t <= now)?.[1] ?? 0;
      lane(name, tokens, total, Math.min(now, marks.at(-1)[0]));
      if (name === "alone") {
        for (const [i, span] of [...$("race-alone-t").children].entries()) span.classList.toggle("on", i < tokens);
      }
    }
    if (now >= end) break;
    await sleep(40);
  }
  const pace = speed === 1 ? "real speed" : `${speed.toFixed(1)} times real speed`;
  $("race-note").textContent =
    `${timing.same ? "The same text" : "Nearly the same text"} written both ways. ` +
    `Every pass was timed on ${timing.device}, and the race plays at ${pace}.`;
}

function resetStats(depth) {
  state.raceId++;
  aloneText([]);
  lane("spec", 0, 0, 0);
  lane("alone", 0, 0, 0);
  $("race-note").textContent = "";
  for (const id of ["s-tok", "s-pass", "s-rate"]) $(id).textContent = "0";
  $("s-per").textContent = $("s-fast").textContent = "0.00";
  const bars = Array.from({ length: depth }, (_, i) => {
    const bar = document.createElement("div");
    bar.dataset.l = i + 1;
    bar.style.height = "2px";
    return bar;
  });
  $("s-hist").replaceChildren(...bars);
  $("d-out").replaceChildren();
}

function showStats(steps, seconds, alone) {
  const written = tokensIn(steps);
  $("s-tok").textContent = written;
  $("s-pass").textContent = steps.length;
  $("s-per").textContent = (written / steps.length).toFixed(2);
  $("s-rate").textContent = seconds ? (written / seconds).toFixed(1) : "n/a";
  $("s-fast").textContent = (alone / seconds).toFixed(2);
  [...$("s-hist").children].forEach((bar, j) => {
    const reached = steps.filter((s) => s.drafted.length > j).length;
    const kept = steps.filter((s) => s.kept > j).length;
    bar.style.height = `${reached ? (36 * kept) / reached : 2}px`;
  });
}

function token(text, className) {
  // A token holding part of a character shows nothing of it, since the token that finishes it shows it whole.
  const node = Object.assign(document.createElement("span"), { textContent: partial(text)[0], className });
  $("d-out").append(node);
  return node;
}

/** Animates one pass: guesses appear, the check colours them, then the misses go and the big model's token lands. */
async function playStep({ drafted, kept, target }, pace) {
  const guesses = [];
  for (const text of drafted) {
    guesses.push(token(text, "guess"));
    if (pace) await sleep(pace * 0.15);
  }
  if (pace) await sleep(pace * 0.6);
  guesses.forEach((node, j) => {
    node.className = j < kept ? "kept" : j === kept ? "miss" : "guess gone";
  });
  if (pace) await sleep(pace);
  for (const node of guesses.slice(kept)) node.remove();
  for (const node of guesses.slice(0, kept)) node.className = "";
  token(target, "own");
}

async function loadModels() {
  status("Loading the tokenizer and runtime...", 0);
  state.ort ??= await runtime();
  state.tok ??= await (await import(LIBRARY)).AutoTokenizer.from_pretrained(TARGET, { revision: TARGET_REVISION });
  if (state.pass) return;
  const sizes = new Map();
  const gb = (n) => (n / 1e9).toFixed(2);
  const file = (url) =>
    cached(url, (loaded, total) => {
      sizes.set(url, [loaded, total]);
      const [got, all] = [...sizes.values()].reduce(([a, b], [l, t]) => [a + l, b + t], [0, 0]);
      status(`Downloading 4-bit weights, ${gb(got)} of ${gb(all)} GB (cached after the first run)`, got / all);
    });
  const [graph, weights, pass, headWeights] = await Promise.all(
    [`${HEAD}/big.onnx`, WEIGHTS, `${HEAD}/pass.onnx`, `${HEAD}/head.onnx.data`].map(file),
  );
  status("Loading the model onto the GPU...", 1);
  const hosted = { path: "model_q4f16.onnx_data", data: weights };
  state.big = await captured(graph, [hosted], 1);
  state.pass = await captured(pass, [hosted, { path: "head.onnx.data", data: headWeights }], 2);
}

/**
 * A session that WebGPU replays from recorded commands, so a pass costs no CPU time per kernel. That needs fixed
 * shapes and every input and output in a GPU buffer, which each pass writes and reads in place: `rows` tokens in,
 * and one cache buffer per layer that is both the past and the present.
 */
async function captured(bytes, externalData, rows) {
  const ort = state.ort;
  // As a plain string the provider never hears about graph capture, though the session still acts as if it did.
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: [{ name: "webgpu" }],
    enableGraphCapture: true,
    logSeverityLevel: 3,
    externalData,
  });
  const device = ort.env.webgpu.device;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const tensor = (dataType, dims, bytesEach) => {
    const size = Math.ceil((dims.reduce((a, b) => a * b, 1) * bytesEach) / 16) * 16;
    return ort.Tensor.fromGpuBuffer(device.createBuffer({ size, usage }), { dataType, dims });
  };
  const shapes = {
    input_ids: ["int64", [1, rows], 8],
    position_ids: ["int64", [1, rows], 8],
    next_ids: ["int64", [1, rows], 8],
    given: ["bool", [1, rows], 1],
    seqlens: ["int32", [1], 4],
    total: ["int32", [], 4],
    wants: ["int64", [1, rows], 8],
    guesses: ["int64", [1, rows], 8],
    logits_fp16: ["float16", [1, rows, 151936], 2],
  };
  const cache = ["float16", [1, 8, ENTRIES, 128], 2];
  const feeds = Object.fromEntries(session.inputNames.map((name) => [name, tensor(...(shapes[name] ?? cache))]));
  // The new cache rows are written into the old cache's buffer.
  const fetches = Object.fromEntries(
    session.outputNames.map((name) => {
      const past = name.replace("present", "past_key_values").replace(/_out$/, "");
      return [name, feeds[past] ?? tensor(...shapes[name])];
    }),
  );
  const s = { session, feeds, fetches, rows, device };
  const buffers = [...Object.values(feeds), ...Object.values(fetches)].map((t) => t.gpuBuffer);
  await pinned(buffers, () => feed(s, Array(rows).fill(0), 0));
  return s;
}

/**
 * Runs `first`, the session's first run, keeping ORT's handles to `buffers` alive after it. ORT web 1.31.1 frees the
 * handle to a buffer passed in at the end of every run and makes a new one the next time, but a captured session
 * keeps using the handles from its first run, which by then point at other buffers. While a run is going, ORT keeps
 * [handle, uses] on each buffer, and setting the uses to Infinity stops it ever reaching 0.
 */
async function pinned(buffers, first) {
  const make = GPUDevice.prototype.createBindGroup;
  GPUDevice.prototype.createBindGroup = function (descriptor) {
    GPUDevice.prototype.createBindGroup = make;
    for (const buffer of buffers) {
      for (const key of Object.getOwnPropertySymbols(buffer)) {
        if (Array.isArray(buffer[key])) buffer[key][1] = Number.POSITIVE_INFINITY;
      }
    }
    return make.call(this, descriptor);
  };
  try {
    await first();
  } finally {
    GPUDevice.prototype.createBindGroup = make;
  }
}

/** Copies GPU buffers back to the CPU in one trip, as the bytes of each. */
async function readBack({ device }, tensors) {
  const sizes = tensors.map((t) => t.gpuBuffer.size);
  const staging = device.createBuffer({
    size: sizes.reduce((a, b) => a + b),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  const starts = sizes.map((_, i) => sizes.slice(0, i).reduce((a, b) => a + b, 0));
  for (const [i, t] of tensors.entries()) encoder.copyBufferToBuffer(t.gpuBuffer, 0, staging, starts[i], sizes[i]);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const all = staging.getMappedRange().slice(0);
  staging.destroy();
  return sizes.map((size, i) => all.slice(starts[i], starts[i] + size));
}

const since = (start) => (performance.now() - start) / 1000;

/**
 * One pass over `ids`, exactly `rows` of them, written into the caches at position `at`. Where `nexts` gives the
 * token after a row, the head reads it in place of the big model's pick. `wants` is the big model's pick after each
 * row and `guesses` the head's.
 */
async function feed(s, ids, at, nexts = []) {
  const put = (name, values) => {
    const bytes = new Uint8Array(Math.ceil(values.byteLength / 4) * 4);
    bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
    s.device.queue.writeBuffer(s.feeds[name].gpuBuffer, 0, bytes);
  };
  put("input_ids", BigInt64Array.from(ids, BigInt));
  put(
    "position_ids",
    BigInt64Array.from(ids, (_, i) => BigInt(at + i)),
  );
  put("seqlens", Int32Array.of(at + ids.length - 1));
  put("total", Int32Array.of(at + ids.length));
  if (s.feeds.given) {
    put(
      "given",
      Uint8Array.from(ids, (_, i) => (nexts[i] === undefined ? 0 : 1)),
    );
    put(
      "next_ids",
      BigInt64Array.from(ids, (_, i) => BigInt(nexts[i] ?? 0)),
    );
  }
  await s.session.run(s.feeds, s.fetches);
  const names = s.fetches.guesses ? ["wants", "guesses"] : ["wants"];
  const [wants, guesses] = await readBack(
    s,
    names.map((name) => s.fetches[name]),
  );
  const ints = (bytes) => (bytes ? Array.from(new BigInt64Array(bytes, 0, s.rows), Number) : []);
  return { wants: ints(wants), guesses: ints(guesses) };
}

/** Reads the prompt `s.rows` tokens a pass, the last pass starting early so it is whole. */
async function read(s, prompt) {
  const starts = [];
  for (let at = 0; at < prompt.length - s.rows; at += s.rows) starts.push(at);
  starts.push(prompt.length - s.rows);
  let got;
  // Rows read twice are written over with the same values, as everything before them is the same.
  for (const at of starts) {
    const ids = prompt.slice(at, at + s.rows);
    got = await feed(
      s,
      ids,
      at,
      ids.map((_, i) => prompt[at + i + 1]),
    );
  }
  return got;
}

/**
 * The loop of switchboard/eagle.py over the pass graph, one guess at a time. Each pass is its guesses, how many
 * were kept, the big model's own token and its seconds. Both ways start their clocks once the prompt is read, which
 * the pass graph does two tokens a pass and the big graph one.
 */
async function guessed(prompt, most, each = () => {}) {
  let got = await read(state.pass, prompt);
  const seq = [...prompt, got.wants.at(-1)];
  const passes = [{ guesses: [], kept: 0, own: seq.at(-1), seconds: 0 }];
  let guesses = got.guesses.slice(-1);
  while (!STOP.has(seq.at(-1)) && seq.length - prompt.length < most) {
    const start = performance.now();
    got = await feed(state.pass, [seq.at(-1), ...guesses], seq.length - 1);
    let kept = accepted(guesses, got.wants);
    // The head guessed after every row, and the one after the last right row is next.
    const after = got.guesses[kept];
    // A kept guess can end the answer, and nothing guessed after it counts.
    const end = guesses.slice(0, kept).findIndex((id) => STOP.has(id));
    if (end >= 0) {
      guesses = guesses.slice(0, end);
      kept = end;
    }
    seq.push(...guesses.slice(0, kept), got.wants[kept]);
    passes.push({ guesses, kept, own: got.wants[kept], seconds: since(start) });
    each(passes);
    guesses = [after];
  }
  return passes;
}

/** The big model's next `n` greedy tokens, one pass each, and the seconds each pass took. */
async function alone(prompt, n, each = () => {}) {
  const seq = [...prompt, (await read(state.big, prompt)).wants[0]];
  const times = [0];
  while (times.length < n) {
    const start = performance.now();
    seq.push((await feed(state.big, seq.slice(-1), seq.length - 1)).wants[0]);
    times.push(since(start));
    each(times);
  }
  return { tokens: seq.slice(prompt.length), times };
}

/** The big model's two best tokens after `ids`, as text with their logits. It wrote the last `said` of them. */
async function topTwo(ids, said) {
  await read(state.big, ids);
  const [bytes] = await readBack(state.big, [state.big.fetches.logits_fp16]);
  const row = Array.from(new Uint16Array(bytes, 0, 151936), half);
  const order = [...row.keys()].sort((a, b) => row[b] - row[a]).slice(0, 2);
  const written = said ? ids.slice(-said) : [];
  return order.map((id) => ({ text: pieces(state.tok, [...written, id]).at(-1), logit: row[id] }));
}

/** An fp16 number from its 16 bits. */
function half(bits) {
  const exponent = (bits >> 10) & 31;
  const fraction = bits & 1023;
  const sign = bits & 32768 ? -1 : 1;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

async function run() {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) {
    status("This browser has no WebGPU, so here is a recorded run. Desktop Chrome, Edge and Safari 26 have it.");
    return replay(true);
  }
  state.replaying = false;
  setBusy(true);
  const depth = GUESSES;
  const most = Number($("d-max").value);
  const text = $("d-prompt").value;
  let trace;
  let said;
  try {
    await loadModels();
    const chat = state.tok.apply_chat_template([{ role: "user", content: text }], {
      tokenize: false,
      add_generation_prompt: true,
      enable_thinking: false,
    });
    const prompt = state.tok.encode(chat, { add_special_tokens: false });
    // The pass graph writes one row past the answer, a guess it may turn down.
    const room = ENTRIES - prompt.length - 2;
    if (room < 16) throw new Error(`the prompt is ${prompt.length} tokens, and this run has room for ${ENTRIES}`);
    resetStats(depth);
    // The first passes of each shape compile shaders, so both ways get a few before the clock counts.
    status("Warming up...", 1);
    await guessed(prompt, 8);
    await alone(prompt, 3);
    const passes = await guessed(prompt, Math.min(most, room), (done) =>
      status(`Writing with guesses on this GPU, ${tokensIn(done)} tokens so far...`),
    );
    const ours = passes.flatMap((pass) => [...pass.guesses.slice(0, pass.kept), pass.own]);
    const solo = await alone(prompt, ours.length, (done) =>
      status(`Writing the same ${ours.length} tokens alone, ${done.length} so far...`),
    );
    const same = accepted(ours, solo.tokens);
    const written = [];
    const steps = passes.map(({ guesses, kept, own, seconds }) => {
      const added = [...guesses.slice(0, kept), own];
      // Spelled after what is already written, so a character split across passes comes out whole.
      const step = {
        drafted: pieces(state.tok, [...written, ...guesses]).slice(written.length),
        kept,
        target: pieces(state.tok, [...written, ...added]).at(-1),
        seconds,
      };
      written.push(...added);
      return step;
    });
    const timing = { alone: solo.times, same: same === ours.length, device: "this GPU" };
    trace = { draft: "EAGLE-3 head", target: "Qwen/Qwen3-1.7B", prompt: text, depth, steps, timing };
    if (timing.same) {
      const total = (times) => times.reduce((a, b) => a + b, 0).toFixed(2);
      said =
        `Written on this GPU just now. The same ${ours.length} tokens took ${secondsIn(steps).toFixed(2)} s with ` +
        `the head guessing one token a pass and ${total(solo.times)} s alone, both after reading the prompt.`;
    } else {
      const [a, b] = await topTwo([...prompt, ...ours.slice(0, same)], same);
      said =
        `The two agree for ${same} tokens, then split where "${a.text}" and "${b.text}" score ${a.logit.toFixed(2)} ` +
        `and ${b.logit.toFixed(2)}. In half precision a one-pass check and token-by-token decoding round slightly ` +
        "differently, so a tie that close can go either way.";
    }
  } catch (error) {
    status(`Couldn't run the models (${error.message}). Here is a recorded run instead.`);
    setBusy(false);
    return replay(true);
  }
  setBusy(false);
  await play(trace, said);
}

async function replay(keepStatus = false) {
  try {
    const trace = await json(TODAY).catch(() => json("data/draft.json"));
    const who = `${trace.target.split("/").pop()}, its ${trace.draft} guessing ${trace.depth} tokens a pass`;
    const { title, link, day } = trace.headline ?? {};
    const said = title
      ? [
          `Today's top Hacker News headline${day ? ` (${day})` : ""}, `,
          Object.assign(document.createElement("a"), { href: link, textContent: title }),
          `, explained by ${who}.`,
        ]
      : `A recorded run by ${who}.`;
    await play(trace, keepStatus ? null : said);
  } catch (error) {
    status(`Couldn't load the recorded run (${error.message}).`);
  }
}

/** Plays each pass in its own measured slot while the alone lane writes on the same clock. */
async function play(trace, said) {
  setReplaying(true);
  try {
    resetStats(trace.depth);
    if (said) status(said, 1);
    const longest = Math.max(
      secondsIn(trace.steps),
      trace.timing.alone.reduce((a, b) => a + b, 0),
    );
    const speed = Math.max(1, longest / RACE);
    const raced = race(trace, speed);
    const started = performance.now();
    let at = 0;
    for (const [i, step] of trace.steps.entries()) {
      // A live run takes over the figure.
      if (!state.replaying) return;
      const wait = (1000 * at) / speed - (performance.now() - started);
      if (!calm && wait > 0) await sleep(wait);
      // playStep spends about 2.2 paces on a pass with 4 guesses.
      await playStep(step, calm ? 0 : (1000 * step.seconds) / speed / 2.2);
      at += step.seconds;
      const done = trace.steps.slice(0, i + 1);
      // What the big model alone took to get as far as the guessed run has by now.
      const alone = trace.timing.alone.slice(0, tokensIn(done)).reduce((a, b) => a + b, 0);
      showStats(done, secondsIn(done), alone);
    }
    await raced;
  } finally {
    setReplaying(false);
  }
}

export function initDraft() {
  $("d-run").addEventListener("click", run);
  $("d-replay").addEventListener("click", () => replay());
  for (const chip of $("d-tries").children) {
    chip.addEventListener("click", () => {
      $("d-prompt").value = chip.dataset.prompt;
      run();
    });
  }
  const watcher = new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting) && !state.busy && !state.replaying && !$("d-out").textContent) {
        watcher.disconnect();
        replay();
      }
    },
    { threshold: 0.4 },
  );
  watcher.observe($("d-out"));
}
