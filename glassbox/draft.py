"""Greedy speculative decoding: Qwen3-0.6B guesses tokens and Qwen3-1.7B checks them in one pass."""

from collections.abc import Sequence
from typing import NamedTuple, TypedDict

from .svg import FAINT, INK, LOST, MONO, SERIF, SHEET, SOFT, Timeline, W, callout, document, line, rect, text

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


def chip(p: Placed, kind: str, cls: str) -> str:
    """A token box drawn the way the page draws it: dashed for a guess, solid if kept, red and struck if missed."""
    box = {
        "guess": rect(p.x - 2, p.y - 27, p.width + 4, 36, SHEET, stroke=INK, dash="5 4"),
        "kept": rect(p.x - 2, p.y - 27, p.width + 4, 36, SHEET, stroke=INK),
        "miss": rect(p.x - 2, p.y - 27, p.width + 4, 36, SHEET, stroke=LOST, width=2.5, dash="5 4"),
    }[kind]
    ink = LOST if kind == "miss" else INK
    strike = line(p.x, p.y - 7, p.x + p.width, p.y - 7, LOST, 2) if kind == "miss" else ""
    return f'<g class="{cls}">{box}{text(p.x, p.y, p.text, 20, ink, font=MONO)}{strike}</g>'


def render(t: Trace) -> str:
    steps, depth = t["steps"], t["depth"]
    tl = Timeline(1.0 + STEP * len(steps) + 3.0)
    end = tl.period - 0.5
    body = [
        text(60, 70, "Speculative Decoding, One Verify Pass at a Time", 34, weight=700, font=SERIF),
        text(
            60,
            106,
            f"Qwen3-0.6B guesses the next {depth} tokens, and Qwen3-1.7B checks all {depth} in a single forward "
            "pass. It keeps the guesses up to the first miss, then adds one token of its own.",
            19,
            SOFT,
        ),
        text(60, 156, f"Prompt: {t['prompt']}", 18, SOFT, font=MONO),
    ]

    # Committed text first. Guesses past a miss sit where later text lands, so padding covers the last pass.
    steps = [
        {**s, "drafted": [g.replace("\n", " ") for g in s["drafted"]], "target": s["target"].replace("\n", " ")}
        for s in steps
    ]
    written = [p for s in steps for p in [*s["drafted"][: s["kept"]], s["target"]]]
    spots = layout(written + [" " * 6] * depth, MARGIN, 218, W - MARGIN)
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
            body.append(chip(p, "guess", hidden[-1]))
            if j <= s["kept"]:
                hidden.append(tl.show(f"v{si}_{j}", verify, settle))
                body.append(chip(p, "kept" if j < s["kept"] else "miss", hidden[-1]))
        for j in range(s["kept"] + 1):
            spot = spots[start + j]
            hidden.append(tl.show(f"w{si}_{j}", settle, end))
            own = j == s["kept"]
            mark = (
                line(spot.x, spot.y + 7, spot.x + spot.width, spot.y + 7, INK, 3) if own and spot.text.strip() else ""
            )
            body.append(
                f'<g class="{hidden[-1]}">'
                f"{text(spot.x, spot.y, spot.text, 20, INK, font=MONO, weight=700 if own else 400)}{mark}</g>"
            )
        start += s["kept"] + 1

    # One bar per big-model pass: solid for kept guesses, outlined for its own token.
    by = max(p.y for p in spots) + 70
    bw = (W - 120) / len(steps)
    body.append(text(60, by, "tokens gained per pass of the big model", 15, SOFT))
    for si, s in enumerate(steps):
        x = 60 + si * bw
        body.append(rect(x, by + 14, bw - 8, (depth + 1) * 18, FAINT))
        cells = [
            rect(x, by + 14 + (depth - j) * 18, bw - 8, 16, INK)
            if j < s["kept"]
            else rect(x + 1, by + 15 + (depth - j) * 18, bw - 10, 14, SHEET, stroke=INK, width=2)
            for j in range(s["kept"] + 1)
        ]
        hidden.append(tl.show(f"b{si}", 1.0 + si * STEP + 1.15, end))
        body.append(f'<g class="{hidden[-1]}">{"".join(cells)}</g>')

    fy = by + 14 + (depth + 1) * 18 + 40
    kx = 60.0
    for draw, numeral, label in [
        (lambda x: rect(x, fy - 15, 22, 16, stroke=INK, dash="4 3"), 102, "guessed by the small model"),
        (lambda x: rect(x, fy - 15, 22, 16, stroke=INK), 104, "kept by the big model"),
        (lambda x: rect(x, fy - 15, 22, 16, stroke=LOST, width=2, dash="4 3"), 106, "first miss, thrown away"),
        (lambda x: line(x, fy, x + 22, fy, INK, 3), 108, "the big model's own token"),
    ]:
        entry, width = callout(kx, fy, draw(kx), numeral, label)
        body.append(entry)
        kx += width + 34
    body.append(
        text(
            60,
            fy + 36,
            f"{len(written)} tokens from {len(steps)} big-model passes, "
            f"{len(written) / len(steps):.2f} per pass instead of 1",
            17,
            INK,
            "start",
            weight=700,
        )
    )
    return document(
        "Speculative decoding: a small model guesses tokens and a big model checks them in one pass",
        int(fy + 76),
        tl.css(",".join(f".{c}" for c in hidden) + "{opacity:0}"),
        body,
    )
