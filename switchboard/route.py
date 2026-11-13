"""Which experts IBM Granite 3.0 1B-A400M's router picks for each token, layer by layer."""

import http.client
import json
import urllib.request
from collections.abc import Sequence
from html.parser import HTMLParser
from itertools import pairwise
from typing import NotRequired, TypedDict

from . import pieces
from .svg import FAINT, INK, MONO, SERIF, SHEET, SOFT, Timeline, W, callout, document, paint, rect, text

MODEL = "ibm-granite/granite-3.0-1b-a400m-instruct"
BROWSER = {"User-Agent": "Mozilla/5.0"}  # many sites turn away a client that doesn't name a browser
SENTENCES = (
    "The kernel adds 128 floats per thread, then def main(): print('done') runs in Python.",
    "Bonjour ! Je voudrais un café et deux croissants, s'il vous plaît.",
    "The integral of x^2 from 0 to 1 equals 1/3, so the area is a third of the unit square.",
    "SELECT name, COUNT(*) FROM orders GROUP BY name HAVING COUNT(*) > 5;",
    "Once upon a time, a small robot learned to paint the sea at night.",
    "Launch day! 🚀 Our app hit 10,000 users in 24 hours 🎉",
)
SLOT, FADE = 0.85, 0.2  # seconds each token stays lit, and seconds it takes to hand over to the next
SWEEP = 0.012  # seconds each layer's picks come up after the layer above's


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
    headline: NotRequired[dict[str, str | int]]  # the daily run's day, Hacker News link and place there


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
        runs.append({"text": sentence, "tokens": pieces(tok, ids[0].tolist()), "layers": list(seen)})
    return {"model": MODEL, "experts": model.config.num_local_experts, "k": k, "runs": runs}


def headline() -> tuple[str, str, int]:
    """The highest of today's top ten Hacker News stories that can be read: its title, discussion page and place.

    Some sites turn a script away, and it is the top story when all ten do.
    """

    def get(url: str) -> dict | list:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)

    tops = get("https://hacker-news.firebaseio.com/v0/topstories.json")[:10]
    links = [f"https://news.ycombinator.com/item?id={top}" for top in tops]
    place = next((at for at, link in enumerate(links) if story(link)), 0)
    title = get(f"https://hacker-news.firebaseio.com/v0/item/{tops[place]}.json")["title"]
    return title, links[place], place + 1


class Paragraphs(HTMLParser):
    """Collects a page's description and the text of its paragraphs."""

    def __init__(self) -> None:
        super().__init__()
        self.found: list[str] = []
        self.inside = self.unseen = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        named = dict(attrs)
        if tag == "meta" and (named.get("name") or named.get("property")) in {"description", "og:description"}:
            self.found.append(named.get("content") or "")
        elif tag == "p":
            self.found.append("")
            self.inside += 1
        elif tag in {"script", "style"}:
            self.unseen += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "p":
            self.inside = max(0, self.inside - 1)
        elif tag in {"script", "style"}:
            self.unseen = max(0, self.unseen - 1)

    def handle_data(self, data: str) -> None:
        if self.inside and not self.unseen:
            self.found[-1] += data


def opening(page: str, most: int = 700) -> str:
    """The first paragraphs of an HTML page, to about `most` characters, or "" when there is too little to go on."""
    parser = Paragraphs()
    parser.feed(page)
    kept: list[str] = []
    for found in parser.found:
        said = " ".join(found.split())
        # Bylines, dates and menus are short, and a description often repeats the first paragraph.
        if len(said) >= 60 and not any(said[:40] in k or k[:40] in said for k in kept):
            kept.append(said)
        if sum(map(len, kept)) >= most:
            break
    said = " ".join(kept)[: most + 200]
    # Cut at the end of a sentence where there is one.
    end = max(said.rfind(". "), said.rfind("? "), said.rfind("! "))
    said = said if len(said) < most or end < most // 2 else said[: end + 1]
    return said if len(said) >= 200 else ""


def story(link: str) -> str:
    """How the story behind a Hacker News discussion page opens, or "" when it can't be read."""
    item = link.rsplit("=", 1)[-1]
    try:
        with urllib.request.urlopen(f"https://hacker-news.firebaseio.com/v0/item/{item}.json", timeout=30) as r:
            told = json.load(r)
        if "url" not in told:
            return opening(f"<p>{told.get('text', '')}</p>")
        with urllib.request.urlopen(urllib.request.Request(told["url"], headers=BROWSER), timeout=30) as r:
            if "html" not in r.headers.get_content_type():
                print(f"Couldn't read the story, as {told['url']} is {r.headers.get_content_type()}")
                return ""
            return opening(r.read(2_000_000).decode(r.headers.get_content_charset() or "utf-8", errors="replace"))
    except (OSError, http.client.HTTPException, ValueError) as error:
        print(f"Couldn't read the story ({error})")
        return ""


def render(t: Trace, subtitle: str = "") -> str:
    run, n, k = t["runs"][0], t["experts"], t["k"]
    tokens, layers = run["tokens"], run["layers"]
    tl = Timeline(SLOT * len(tokens) + 1.5)
    # Neighbouring tokens overlap by the fade, so one set of experts dims as the next lights.
    lit = [
        tl.show(f"t{i}", 0.6 + i * SLOT - FADE / 2, 0.6 + (i + 1) * SLOT + FADE / 2, fade=FADE)
        for i in range(len(tokens))
    ]
    subtitle = subtitle or (
        f"IBM Granite 3.0 1B-A400M has {n} experts in each of its {len(layers)} layers, and its router sends every "
        f"token to {k} of them. This is a real forward pass over one sentence."
    )
    body = [
        text(60, 70, "How a Mixture-of-Experts Model Routes Each Token", 34, weight=700, font=SERIF),
        text(60, 106, subtitle, 19, SOFT),
    ]

    x, y = 60.0, 164.0
    for i, s in enumerate(tokens):
        shown = s.replace(" ", "·") if s.strip() else "·"
        w = 11.2 * len(shown) + 14
        if x + w > W - 60:
            x, y = 60.0, y + 46
        body += [
            rect(x, y - 26, w, 38, stroke=FAINT),
            rect(x, y - 26, w, 38, INK, lit[i]),
            text(x + 7, y, shown, 18, INK, font=MONO),
            text(x + 7, y, shown, 18, SHEET, font=MONO, weight=700, cls=lit[i]),
        ]
        x += w + 6

    # The faint wash is the sentence's total router weight per expert.
    gx, gy, cw, ch = 110, y + 70, 34.0, 19.0
    body += [text(gx + e * cw, gy - 14, str(e), 14, SOFT) for e in range(0, n, 8)]
    body.append(text(gx + n * cw - 3, gy - 14, str(n - 1), 14, SOFT, "end"))
    use = [[0.0] * n for _ in layers]
    for li, layer in enumerate(layers):
        for row, weights in zip(layer["experts"], layer["weights"], strict=True):
            for e, wt in zip(row, weights, strict=True):
                use[li][e] += wt
    peak = max(map(max, use))
    for li, layer in enumerate(layers):
        y = gy + li * ch
        if li % 4 == 0 or li == len(layers) - 1:
            body.append(text(gx - 14, y + 14, str(li), 14, SOFT, "end"))
        for e in range(n):
            heat = f' fill-opacity="{0.03 + 0.3 * use[li][e] / peak:.3f}"'
            body += [
                rect(gx + e * cw, y, cw - 3, ch - 3, stroke=FAINT, width=1),
                rect(gx + e * cw, y, cw - 3, ch - 3, INK, extra=heat),
            ]
        for ti, (row, weights) in enumerate(zip(layer["experts"], layer["weights"], strict=True)):
            before = set(layer["experts"][ti - 1]) if ti else set()
            # The layer's class delays its picks, so a token's experts come up in the order the pass picks them.
            cls = f"{lit[ti]} l{li}"
            for e, wt in zip(row, weights, strict=True):
                strength = f' fill-opacity="{0.45 + 0.55 * min(1, wt * 3):.2f}"'
                body.append(rect(gx + e * cw, y, cw - 3, ch - 3, INK, cls=cls, extra=strength))
                if e in before:
                    body.append(dot(gx + e * cw + (cw - 3) / 2, y + (ch - 3) / 2, cls))

    sx = gx + n * cw + 50
    for j, (swatch, numeral, label, more) in enumerate(
        [
            (rect(sx, gy + 6, 18, 18, INK), 202, "an expert this token", "goes to, stronger for more weight"),
            (rect(sx, gy + 72, 18, 18, INK) + dot(sx + 9, gy + 81), 204, "also used by", "the token before"),
            (
                rect(sx, gy + 6 + 132, 18, 18, INK, extra=' fill-opacity="0.15"'),
                206,
                "the sentence's total",
                "weight on that expert",
            ),
        ]
    ):
        entry, _ = callout(sx, gy + 20 + 66 * j, swatch, numeral, label)
        body += [entry, text(sx + 68, gy + 44 + 66 * j, more, 17)]
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
    body += [text(sx, gy + 250 + 27 * j, line, 17, SOFT) for j, line in enumerate(lines)]
    # These come after the timeline's rules, whose `animation` shorthand would set the delay back to none.
    delays = "".join(f".l{li}{{animation-delay:{li * SWEEP:g}s}}" for li in range(1, len(layers)))
    return document(
        "A mixture-of-experts router picking experts for each token, layer by layer",
        int(gy + len(layers) * ch + 40),
        tl.css(",".join(f".{c}" for c in lit) + "{opacity:0}") + "\n" + delays,
        body,
    )


def dot(x: float, y: float, cls: str = "") -> str:
    """The mark on an expert that the token before went to as well."""
    c = f' class="{cls}"' if cls else ""
    return f'<circle{c} cx="{x:.1f}" cy="{y:.1f}" r="3"{paint(SHEET)}/>'
