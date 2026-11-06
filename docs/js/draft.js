import { $, calm, json, sleep } from "./dom.js";

const LIBRARY = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";
const DRAFT = "onnx-community/Qwen3-0.6B-ONNX";
// The same 4-bit export as onnx-community/Qwen3-1.7B-ONNX with its weights in a separate data file. The single
// 1.4 GB file runs out of WebAssembly memory while the session is created.
const TARGET = "Exempliss/Qwen3-1.7B-ONNX-Split";
const TARGET_REVISION = "f9b9656836a4315a690d0270f885b0758923bd13";
const STOP = new Set([151643, 151645]); // Qwen3 <|endoftext|> and <|im_end|>

const state = {
  lib: null,
  tok: null,
  small: null,
  big: null,
  busy: false,
  replaying: false,
  ids: null,
  promptLength: 0,
};

/** How many guesses match the big model's greedy picks before the first miss. */
export function accepted(guesses, wants) {
  let kept = 0;
  while (kept < guesses.length && guesses[kept] === wants[kept]) kept++;
  return kept;
}

function status(message, progress) {
  $("d-status").textContent = message;
  if (progress !== undefined) $("d-bar").style.transform = `scaleX(${progress})`;
}

function setBusy(busy) {
  state.busy = busy;
  $("d-run").disabled = $("d-replay").disabled = busy;
  $("d-check").disabled = busy || !state.ids;
}

function setReplaying(replaying) {
  state.replaying = replaying;
  $("d-replay").disabled = replaying;
}

function resetStats(depth) {
  for (const id of ["s-tok", "s-pass", "s-rate"]) $(id).textContent = "0";
  $("s-per").textContent = "0.00";
  const bars = Array.from({ length: depth }, (_, i) => {
    const bar = document.createElement("div");
    bar.dataset.l = i + 1;
    bar.style.height = "2px";
    return bar;
  });
  $("s-hist").replaceChildren(...bars);
  $("d-out").replaceChildren();
}

function showStats(steps, seconds) {
  const written = steps.reduce((sum, s) => sum + s.kept + 1, 0);
  $("s-tok").textContent = written;
  $("s-pass").textContent = steps.length;
  $("s-per").textContent = (written / steps.length).toFixed(2);
  $("s-rate").textContent = seconds ? (written / seconds).toFixed(1) : "n/a";
  [...$("s-hist").children].forEach((bar, j) => {
    const reached = steps.filter((s) => s.drafted.length > j).length;
    const kept = steps.filter((s) => s.kept > j).length;
    bar.style.height = `${reached ? (36 * kept) / reached : 2}px`;
  });
}

function token(text, className) {
  const node = Object.assign(document.createElement("span"), { textContent: text, className });
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
  status("Loading the inference library...", 0);
  state.lib ??= await import(LIBRARY);
  const files = new Map();
  const progress_callback = (p) => {
    if (p.status !== "progress" || !p.total) return;
    files.set(p.name + p.file, [p.loaded, p.total]);
    const [loaded, total] = [...files.values()].reduce(([a, b], [l, t]) => [a + l, b + t], [0, 0]);
    const gb = (n) => (n / 1e9).toFixed(2);
    status(`Downloading 4-bit weights, ${gb(loaded)} of ${gb(total)} GB (cached after the first run)`, loaded / total);
  };
  const options = { dtype: "q4f16", device: "webgpu", progress_callback };
  state.tok ??= await state.lib.AutoTokenizer.from_pretrained(DRAFT);
  state.big ??= await state.lib.AutoModelForCausalLM.from_pretrained(TARGET, {
    ...options,
    revision: TARGET_REVISION,
    use_external_data_format: true,
  });
  state.small ??= await state.lib.AutoModelForCausalLM.from_pretrained(DRAFT, options);
}

function inputs(ids) {
  const { Tensor } = state.lib;
  return {
    input_ids: new Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]),
    attention_mask: new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
  };
}

/** The big model's greedy pick at every position from `from` on, all from one forward pass. */
async function check(ids, from) {
  const { logits } = await state.big(inputs(ids));
  const vocab = logits.dims[2];
  const data = logits.data;
  const picks = [];
  for (let p = from; p < ids.length; p++) {
    let best = Number.NEGATIVE_INFINITY;
    let pick = 0;
    for (let j = 0, o = p * vocab; j < vocab; j++) {
      if (data[o + j] > best) [best, pick] = [data[o + j], j];
    }
    picks.push(pick);
  }
  logits.dispose?.();
  return picks;
}

/** The big model's two best next tokens after `ids`, with their logits. */
async function topTwo(ids) {
  const { logits } = await state.big(inputs(ids));
  const vocab = logits.dims[2];
  const row = logits.data.subarray((ids.length - 1) * vocab, ids.length * vocab);
  const order = [...row.keys()].sort((a, b) => row[b] - row[a]).slice(0, 2);
  logits.dispose?.();
  return order.map((id) => ({ text: state.tok.decode([id]), logit: Number(row[id]) }));
}

async function guess(ids, depth) {
  const out = await state.small.generate({ ...inputs(ids), max_new_tokens: depth, do_sample: false });
  return out.tolist()[0].slice(ids.length).map(Number);
}

async function run() {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) {
    status("This browser has no WebGPU, so here is a recorded run. Desktop Chrome, Edge and Safari 26 have it.");
    return replay(true);
  }
  state.replaying = false;
  setBusy(true);
  const depth = Number($("d-depth").value);
  const max = Number($("d-max").value);
  try {
    await loadModels();
    const chat = state.tok.apply_chat_template([{ role: "user", content: $("d-prompt").value }], {
      tokenize: false,
      add_generation_prompt: true,
      enable_thinking: false,
    });
    let ids = state.tok.encode(chat, { add_special_tokens: false });
    state.promptLength = ids.length;
    resetStats(depth);
    status("Running. The small model guesses, then the big model checks every guess in one pass.", 1);
    const steps = [];
    const started = performance.now();
    while (ids.length - state.promptLength < max) {
      const guesses = await guess(ids, depth);
      const wants = await check(ids.concat(guesses), ids.length - 1);
      const kept = accepted(guesses, wants);
      const added = [...guesses.slice(0, kept), wants[kept]];
      const step = {
        drafted: guesses.map((id) => state.tok.decode([id], { skip_special_tokens: true })),
        kept,
        target: state.tok.decode([wants[kept]], { skip_special_tokens: true }),
      };
      steps.push(step);
      await playStep(step, calm ? 0 : 160);
      ids = ids.concat(added);
      showStats(steps, (performance.now() - started) / 1000);
      if (added.some((id) => STOP.has(id))) break;
    }
    state.ids = ids;
    status(
      `Done. ${ids.length - state.promptLength} tokens from ${steps.length} big-model passes. Check it against the big model writing alone.`,
    );
  } catch (error) {
    status(`Couldn't run the models (${error.message}). Here is a recorded run instead.`);
    setBusy(false);
    return replay(true);
  }
  setBusy(false);
}

async function compare() {
  setBusy(true);
  try {
    const ours = state.ids.slice(state.promptLength);
    status("Running the big model alone with plain greedy decoding...");
    const out = await state.big.generate({
      ...inputs(state.ids.slice(0, state.promptLength)),
      max_new_tokens: ours.length,
      do_sample: false,
    });
    const alone = out.tolist()[0].slice(state.promptLength).map(Number);
    const same = accepted(ours, alone);
    if (same === ours.length) {
      status(
        `Identical. All ${same} tokens match what Qwen3-1.7B writes alone, which took ${same} big-model passes instead of ${$("s-pass").textContent}.`,
      );
    } else {
      const [a, b] = await topTwo(state.ids.slice(0, state.promptLength + same));
      status(
        `The two agree for ${same} tokens, then split where "${a.text}" and "${b.text}" score ${a.logit.toFixed(2)} and ` +
          `${b.logit.toFixed(2)}. In half precision a one-pass check and token-by-token decoding round slightly ` +
          "differently, so a tie that close can go either way.",
      );
    }
  } catch (error) {
    status(`Couldn't run the check (${error.message}).`);
  } finally {
    setBusy(false);
  }
}

async function replay(keepStatus = false) {
  setReplaying(true);
  try {
    const trace = await json("data/draft.json");
    resetStats(trace.depth);
    $("d-prompt").value = trace.prompt;
    if (!keepStatus) {
      status(`A recorded run of ${trace.target} checking ${trace.draft} in BF16, ${trace.depth} guesses a pass.`, 1);
    }
    for (const [i, step] of trace.steps.entries()) {
      // A live run takes over the figure.
      if (!state.replaying) return;
      await playStep(step, calm ? 0 : 260);
      showStats(trace.steps.slice(0, i + 1), 0);
    }
  } catch (error) {
    status(`Couldn't load the recorded run (${error.message}).`);
  } finally {
    setReplaying(false);
  }
}

export function initDraft() {
  $("d-depth").addEventListener("input", (e) => {
    $("d-depth-v").textContent = e.target.value;
  });
  $("d-run").addEventListener("click", run);
  $("d-replay").addEventListener("click", () => replay());
  $("d-check").addEventListener("click", compare);
  // Play the recorded run once when the panel first scrolls into view.
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
