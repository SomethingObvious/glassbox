"""Greedy speculative decoding: Qwen3-0.6B guesses tokens and Qwen3-1.7B checks them in one pass."""

from collections.abc import Sequence
from typing import NamedTuple, TypedDict

from .svg import AMBER, BG, CYAN, GREEN, INK, LINE, MONO, MUTED, RED, Timeline, W, document, rect, text

DRAFT, TARGET = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"
PROMPT = "In three sentences, explain why the sky is blue."
DEPTH, STEPS = 4, 16
STEP, CHAR, LINE_H, MARGIN = 1.5, 12.0, 46, 60  # s per verify pass, px per 20 px mono character, px per line


class Step(TypedDict):
    drafted: list[str]  # the small model's guesses
    kept: int  # how many of them the big model agreed with
    target: str  # the big model's own token after the kept guesses


class Trace(TypedDict):
    draft: str
    target: str
    prompt: str
    depth: int
    steps: list[Step]


class Placed(NamedTuple):
    x: float
    y: float
    width: float
    text: str


def accepted(guesses: Sequence[int], wants: Sequence[int]) -> int:
    """How many guesses match the big model's greedy picks before the first miss."""
    kept = 0
    while kept < len(guesses) and guesses[kept] == wants[kept]:
        kept += 1
    return kept


def trace() -> Trace:
    """Runs STEPS draft-and-verify passes in BF16 on MPS or CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TARGET)
    small, big = (AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16).to(dev).eval() for m in (DRAFT, TARGET))
    chat = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tok(chat, return_tensors="pt").input_ids[0].to(dev)
    steps: list[Step] = []
    with torch.inference_mode():
        for _ in range(STEPS):
            guesses = small.generate(ids[None], max_new_tokens=DEPTH, do_sample=False)[0, len(ids) :]
            # One big-model pass scores the position after every guess at once.
            wants = big(torch.cat([ids, guesses])[None]).logits[0, len(ids) - 1 :].argmax(-1)
            kept = accepted(guesses.tolist(), wants.tolist())
            steps.append(
                {
                    "drafted": [tok.decode([i]) for i in guesses.tolist()],
                    "kept": kept,
                    "target": tok.decode([wants[kept]]),
                }
            )
            ids = torch.cat([ids, guesses[:kept], wants[kept : kept + 1]])
    return {"draft": DRAFT, "target": TARGET, "prompt": PROMPT, "depth": DEPTH, "steps": steps}


def layout(pieces: Sequence[str], x0: float, y0: float, right: float) -> list[Placed]:
    """Places token strings left to right and wraps them at `right`, dropping the space that starts a line."""
    out, x, y = [], x0, y0
    for piece in pieces:
        wrap = x + CHAR * len(piece) > right and x > x0
        if wrap:
            x, y = x0, y + LINE_H
        shown = piece.lstrip() if wrap else piece
        out.append(Placed(x, y, CHAR * len(shown), shown))
        x += CHAR * len(shown)
    return out


def chip(p: Placed, fill: str, ink: str, cls: str, edge: str = "") -> str:
    stroke = f' stroke="{edge}" stroke-width="2"' if edge else ""
    return (
        f'<g class="{cls}">{rect(p.x - 1, p.y - 27, p.width + 2, 36, fill, 5, extra=stroke)}'
        + text(p.x, p.y, p.text, 20, ink, font=MONO, weight=600)
        + "</g>"
    )


def render(t: Trace) -> str:
    steps, depth = t["steps"], t["depth"]
    tl = Timeline(1.0 + STEP * len(steps) + 3.0)
    end = tl.period - 0.5
    body = [
        text(60, 66, "Speculative decoding, one verify pass at a time", 34, weight=600),
        text(
            60,
            102,
            f"Qwen3-0.6B guesses the next {depth} tokens, and Qwen3-1.7B checks all {depth} in a single forward "
            "pass. It keeps the guesses up to the first miss, then adds one token of its own.",
            19,
            MUTED,
        ),
        text(60, 152, f"Prompt: {t['prompt']}", 18, MUTED, font=MONO),
    ]

    # Committed text first. Guesses past a miss sit where later text lands, so padding covers the last pass.
    steps = [
        {**s, "drafted": [g.replace("\n", " ") for g in s["drafted"]], "target": s["target"].replace("\n", " ")}
        for s in steps
    ]
    written = [p for s in steps for p in [*s["drafted"][: s["kept"]], s["target"]]]
    spots = layout(written + [" " * 6] * depth, MARGIN, 214, W - MARGIN)
    hidden: list[str] = []
    start = 0
    for si, s in enumerate(steps):
        t0 = 1.0 + si * STEP
        verify, settle = t0 + 0.6, t0 + 1.15
        for j, guess in enumerate(s["drafted"]):
            spot = spots[start + j]
            shown = guess.lstrip() if spot.x == MARGIN else guess
            p = Placed(spot.x, spot.y, CHAR * len(shown), shown)
            hidden.append(tl.show(f"g{si}_{j}", t0 + 0.08 * j, verify))
            body.append(chip(p, "#10202a", CYAN, hidden[-1], CYAN))
            if j <= s["kept"]:
                hidden.append(tl.show(f"v{si}_{j}", verify, settle))
                body.append(chip(p, GREEN if j < s["kept"] else RED, BG, hidden[-1]))
        for j in range(s["kept"] + 1):
            spot = spots[start + j]
            hidden.append(tl.show(f"w{si}_{j}", settle, end))
            body.append(
                text(spot.x, spot.y, spot.text, 20, AMBER if j == s["kept"] else INK, font=MONO, cls=hidden[-1])
            )
        start += s["kept"] + 1

    # One bar per big-model pass: green for kept guesses, amber for its own token.
    by = max(p.y for p in spots) + 70
    bw = (W - 120) / len(steps)
    body.append(text(60, by, "tokens gained per pass of the big model", 15, MUTED, font=MONO))
    for si, s in enumerate(steps):
        x = 60 + si * bw
        body.append(rect(x, by + 14, bw - 8, (depth + 1) * 18, LINE, 4))
        cells = [
            rect(x, by + 14 + (depth - j) * 18, bw - 8, 16, GREEN if j < s["kept"] else AMBER)
            for j in range(s["kept"] + 1)
        ]
        hidden.append(tl.show(f"b{si}", 1.0 + si * STEP + 1.15, end))
        body.append(f'<g class="{hidden[-1]}">{"".join(cells)}</g>')

    fy = by + 14 + (depth + 1) * 18 + 40
    kx = 60.0
    for color, label in [
        (CYAN, "guessed by the small model"),
        (GREEN, "kept by the big model"),
        (RED, "first miss"),
        (AMBER, "the big model's own token"),
    ]:
        body += [rect(kx, fy - 14, 16, 16, color), text(kx + 24, fy, label, 17, MUTED)]
        kx += 24 + 9.0 * len(label) + 40
    body.append(
        text(
            W - 60,
            fy,
            f"{len(written)} tokens from {len(steps)} big-model passes, "
            f"{len(written) / len(steps):.2f} per pass instead of 1",
            17,
            INK,
            "end",
            weight=600,
        )
    )
    return document(
        "Speculative decoding: a small model guesses tokens and a big model checks them in one pass",
        int(fy + 34),
        tl.css(",".join(f".{c}" for c in hidden) + "{opacity:0}"),
        body,
    )
