"""Which experts IBM Granite 3.0 1B-A400M's router picks for each token, layer by layer."""

import json
import urllib.request
from collections.abc import Sequence
from itertools import pairwise
from typing import TypedDict

from .svg import AMBER, BG, CYAN, INK, LINE, MONO, MUTED, Timeline, W, document, rect, text

MODEL = "ibm-granite/granite-3.0-1b-a400m-instruct"
SENTENCES = (
    "The kernel adds 128 floats per thread, then def main(): print('done') runs in Python.",
    "Bonjour ! Je voudrais un café et deux croissants, s'il vous plaît.",
    "The integral of x^2 from 0 to 1 equals 1/3, so the area is a third of the unit square.",
    "SELECT name, COUNT(*) FROM orders GROUP BY name HAVING COUNT(*) > 5;",
    "Once upon a time, a small robot learned to paint the sea at night.",
    "Launch day! 🚀 Our app hit 10,000 users in 24 hours 🎉",
)
SLOT = 0.85  # seconds each token stays lit


class Layer(TypedDict):
    experts: list[list[int]]  # per token, the router's top-k experts
    weights: list[list[float]]  # per token, softmax over those k router logits


class Run(TypedDict):
    text: str
    tokens: list[str]
    layers: list[Layer]


class Trace(TypedDict):
    model: str
    experts: int
    k: int
    runs: list[Run]


def trace(sentences: Sequence[str] = SENTENCES) -> Trace:
    """Runs one forward pass per sentence and records every router decision."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    k = model.config.num_experts_per_tok
    seen: list[Layer] = []

    def hook(_module: torch.nn.Module, _inputs: tuple, logits: torch.Tensor) -> None:
        top, experts = torch.topk(logits.float(), k, dim=-1)
        weights = [[round(w, 4) for w in row] for row in torch.softmax(top, -1).tolist()]
        seen.append({"experts": experts.tolist(), "weights": weights})

    for layer in model.model.layers:
        layer.block_sparse_moe.router.layer.register_forward_hook(hook)
    runs: list[Run] = []
    for sentence in sentences:
        seen.clear()
        ids = tok(sentence, return_tensors="pt").input_ids
        with torch.inference_mode():
            model(ids)
        runs.append({"text": sentence, "tokens": [tok.decode([i]) for i in ids[0].tolist()], "layers": list(seen)})
    return {"model": MODEL, "experts": model.config.num_local_experts, "k": k, "runs": runs}


def headline() -> str:
    """Today's top Hacker News story title."""

    def get(url: str) -> dict | list:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)

    top = get("https://hacker-news.firebaseio.com/v0/topstories.json")[0]
    return get(f"https://hacker-news.firebaseio.com/v0/item/{top}.json")["title"]


def render(t: Trace, subtitle: str = "") -> str:
    run, n, k = t["runs"][0], t["experts"], t["k"]
    tokens, layers = run["tokens"], run["layers"]
    tl = Timeline(SLOT * len(tokens) + 1.5)
    lit = [tl.show(f"t{i}", 0.6 + i * SLOT, 0.6 + (i + 1) * SLOT) for i in range(len(tokens))]
    subtitle = subtitle or (
        f"IBM Granite 3.0 1B-A400M has {n} experts in each of its {len(layers)} layers, and its router sends every "
        f"token to {k} of them. This is a real forward pass over one sentence."
    )
    body = [
        text(60, 66, "How a mixture-of-experts model routes each token", 34, weight=600),
        text(60, 102, subtitle, 19, MUTED),
    ]

    # The sentence as tokens, each highlighted while it is the one being routed.
    x, y = 60.0, 160.0
    for i, s in enumerate(tokens):
        shown = s.replace(" ", "·") if s.strip() else "·"
        w = 11.2 * len(shown) + 14
        if x + w > W - 60:
            x, y = 60.0, y + 46
        body += [
            rect(x, y - 26, w, 38, "#121925", 6, extra=f' stroke="{LINE}"'),
            rect(x, y - 26, w, 38, AMBER, 6, lit[i]),
            text(x + 7, y, shown, 18, INK, font=MONO),
            text(x + 7, y, shown, 18, BG, font=MONO, weight=700, cls=lit[i]),
        ]
        x += w + 6

    # Layers down, experts across. Cyan is the sentence's total router weight per expert.
    gx, gy, cw, ch = 150, y + 70, 34.0, 19.0
    body += [
        text(gx, gy - 14, "expert 0", 14, MUTED, font=MONO),
        text(gx + n * cw - 4, gy - 14, f"expert {n - 1}", 14, MUTED, "end", font=MONO),
    ]
    use = [[0.0] * n for _ in layers]
    for li, layer in enumerate(layers):
        for row, weights in zip(layer["experts"], layer["weights"], strict=True):
            for e, wt in zip(row, weights, strict=True):
                use[li][e] += wt
    peak = max(map(max, use))
    for li, layer in enumerate(layers):
        y = gy + li * ch
        body.append(text(gx - 14, y + 14, f"layer {li}", 13, MUTED, "end", font=MONO))
        for e in range(n):
            heat = f' fill-opacity="{0.06 + 0.5 * use[li][e] / peak:.3f}"'
            body.append(rect(gx + e * cw, y, cw - 3, ch - 3, CYAN, extra=heat))
        for ti, (row, weights) in enumerate(zip(layer["experts"], layer["weights"], strict=True)):
            for e, wt in zip(row, weights, strict=True):
                strength = f' fill-opacity="{0.35 + 0.65 * min(1, wt * 3):.2f}"'
                body.append(rect(gx + e * cw, y, cw - 3, ch - 3, AMBER, cls=lit[ti], extra=strength))

    # The key, then two numbers measured from this trace.
    sx = gx + n * cw + 50
    key = [
        (AMBER, "the experts this token goes to,"),
        ("", "brighter for a bigger weight"),
        (CYAN, "how much the whole sentence"),
        ("", "leaned on each expert"),
    ]
    for j, (swatch, line) in enumerate(key):
        yy = gy + 20 + j * 28 + (12 if j >= 2 else 0)
        if swatch:
            body.append(rect(sx, yy - 14, 16, 16, swatch))
        body.append(text(sx + 26, yy, line, 17, MUTED))
    shared = [len(set(a) & set(b)) for layer in layers for a, b in pairwise(layer["experts"])]
    woken = sum(len({e for row in layer["experts"] for e in row}) for layer in layers) / len(layers)
    lines = [
        f"{k} of {n} experts run per layer,",
        f"so {100 * k / n:.0f}% of the expert weights",
        "do the work for any one token.",
        "",
        "Neighbouring tokens share only",
        f"{sum(shared) / len(shared):.1f} of their {k} experts, so this",
        f"{len(tokens)}-token sentence wakes up",
        f"{woken:.1f} of the {n} in a layer on average.",
    ]
    body += [text(sx, gy + 190 + 27 * j, line, 17, INK) for j, line in enumerate(lines)]
    return document(
        "A mixture-of-experts router picking experts for each token, layer by layer",
        int(gy + len(layers) * ch + 40),
        tl.css(",".join(f".{c}" for c in lit) + "{opacity:0}"),
        body,
    )
