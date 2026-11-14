"""Greedy speculative decoding: an EAGLE-3 head guesses tokens and Qwen3-1.7B checks them in one pass."""

import itertools
import re
from collections.abc import Sequence
from typing import NamedTuple, NotRequired, TypedDict

from . import pieces
from .svg import FAINT, INK, LOST, MONO, SERIF, SHEET, SOFT, Timeline, W, callout, document, line, rect, text

DRAFT, TARGET = "EAGLE-3 head", "Qwen/Qwen3-1.7B"
HEAD = {
    "repo_id": "hoyporti/qwen3-1.7b-eagle3-switchboard",
    "filename": "model.safetensors",
    "revision": "efcba5ef85bd4bb69b9fe0577dbd859e6e748e38",
}
PROMPT = "In three sentences, explain why the sky is blue."
DEPTH, MOST, RUNS = 8, 160, 3
SURE = 0.4  # the head stops guessing once its odds that every guess so far is right fall below this
CHAR, LINE_H, MARGIN = 12.0, 46, 60  # px per 20 px mono character, px per line
SMALL, SMALL_H = 9.0, 26  # the same for the 15 px text of the run alone
LEAD, HOLD, OUT = 1.0, 4.0, 0.5  # seconds before the banner starts writing, after it finishes, and to fade out
RACE = 20.0  # most seconds the banner takes to write before it plays faster than real time
WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|')*|\d[\d,.]*")
# Words that start a sentence with a capital letter and name nothing.
OPENERS = frozenset(
    WORD.findall(
        "what why how who whom whose when where which is are was were does do did will can could would should has "
        "have the a an it its this that these those they their he his she her we our you there here in on at by for "
        "with from to as of after before during since while although though because if but and or so however also "
        "some many most one two three both each all no not"
    )
)


class Step(TypedDict):
    drafted: list[str]
    kept: int  # how many of them the big model agreed with
    target: str  # the big model's own token after the kept guesses
    seconds: float  # the whole pass, guessing included


class Timing(TypedDict):
    alone: list[float]  # seconds for each token when the big model writes the same text on its own
    same: bool  # whether both wrote the same tokens, as greedy decoding should
    device: str


class Trace(TypedDict):
    draft: str
    target: str
    prompt: str
    depth: int
    steps: list[Step]
    timing: Timing
    headline: NotRequired[dict[str, str | int]]


def read(title: str, opening: str) -> str:
    # Asked to explain a headline alone, a model this small makes the story up, so it is given the story's opening.
    return (
        f'Rewrite the text below as the first three sentences of a news article headlined "{title}". Begin '
        f"with who or what the news is about, and use only facts from the text.\n\n{opening}"
    )


def ask(title: str) -> str:
    """What the model is asked on a day the story can't be read."""
    return (
        f'You have not read the story behind this Hacker News headline: "{title}"\n'
        "Write three short questions you would want it to answer, one per line. Name nothing the headline does "
        "not name."
    )


def invented(shown: str, answer: str) -> list[str]:
    """The names and numbers in `answer` that `shown` doesn't have, which the model can only have made up.

    A name is a capitalised word, unless it opens a sentence and is one of the common words that do.
    """
    # ponytail: capital letters and digits are all it looks at, so something made up in lower case gets past it.

    def bare(word: str) -> str:
        return word.lower().removesuffix("'s").rstrip("'.,")

    known = {bare(w) for w in WORD.findall(shown.replace("’", "'"))}
    found = []
    for sentence in re.split(r"(?<=[.?!])\s+|\n+", answer.replace("’", "'")):
        words = WORD.findall(re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", sentence))
        for at, word in enumerate(words):
            named = word[0].isdigit() or (word[0].isupper() and (at > 0 or word.lower() not in OPENERS))
            if named and word != "I" and bare(word) not in known:
                found.append(word)
    return found


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


def trace(prompt: str = PROMPT) -> Trace:
    """Writes an answer with guesses and again alone, timing every pass, and keeps the runs with the median total.

    CPUs run in FP32 with packed weights, since those without native BF16 maths run it slower, and anything else in
    BF16. Both ways use the same model, so the race is fair.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import eagle

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if dev == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(TARGET)
    big = AutoModelForCausalLM.from_pretrained(TARGET, dtype=dtype).to(dev).eval()
    if dev == "cpu":
        eagle.pack(big, DEPTH + 1)
    head = eagle.Head(load_file(hf_hub_download(**HEAD)), big.model.embed_tokens)
    chat = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tok(chat).input_ids
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}

    def median(runs: list[list[float]]) -> list[float]:
        """The run with the median total, with both clocks started once the prompt is read, as on the page."""
        return [0.0, *sorted((r[1:] for r in runs), key=sum)[len(runs) // 2]]

    with torch.inference_mode():
        passes = eagle.guessed(big, head, ids, DEPTH, MOST, stop, SURE)  # the first run of each warms up
        n = sum(kept + 1 for _, kept, _, _ in passes)
        eagle.alone(big, ids, n)
        spec, solo = [], []
        for _ in range(RUNS):
            spec.append([s for *_, s in eagle.guessed(big, head, ids, DEPTH, MOST, stop, SURE)])
            written, times = eagle.alone(big, ids, n)
            solo.append(times)
    ours = [t for guesses, kept, own, _ in passes for t in [*guesses[:kept], own]]

    steps: list[Step] = []
    said: list[int] = []
    for (guesses, kept, own, _), seconds in zip(passes, median(spec), strict=True):
        # Spelled after what is already written, so a character split across passes comes out whole.
        added = [*guesses[:kept], own]
        steps.append(
            {
                "drafted": [
                    "" if g in stop else piece
                    for g, piece in zip(guesses, pieces(tok, said + guesses)[len(said) :], strict=True)
                ],
                "kept": kept,
                "target": "" if own in stop else pieces(tok, said + added)[-1],
                "seconds": round(seconds, 4),
            }
        )
        said += added
    device = torch.cuda.get_device_name() if dev == "cuda" else f"{torch.get_num_threads()} CPU cores"
    return {
        "draft": DRAFT,
        "target": TARGET,
        "prompt": prompt,
        "depth": DEPTH,
        "steps": steps,
        "timing": {"alone": [round(x, 4) for x in median(solo)], "same": written == ours, "device": device},
    }


def clip(s: str, most: int) -> str:
    return s if len(s) <= most else s[: most - 3].rstrip() + "..."


def layout(
    pieces: Sequence[str],
    x0: float,
    y0: float,
    right: float,
    char: float = CHAR,
    line_h: float = LINE_H,
    at: float | None = None,
) -> list[Placed]:
    """Places token strings from `at` (or `x0`) and wraps them at `right`, dropping the space that starts a line."""
    out, x, y = [], x0 if at is None else at, y0
    for piece in pieces:
        if x + char * len(piece) > right and x > x0:
            x, y = x0, y + line_h
        shown = piece.lstrip() if x == x0 else piece
        out.append(Placed(x, y, char * len(shown), shown))
        x += char * len(shown)
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
    return f'<g class="{cls}">{box}{text(p.x, p.y, p.text, 20, ink, font=MONO, fit=p.width)}{strike}</g>'


def render(t: Trace) -> str:
    steps, depth, timing = t["steps"], t["depth"], t["timing"]
    spec = list(itertools.accumulate(s["seconds"] for s in steps))
    alone = list(itertools.accumulate(timing["alone"]))
    scale = min(1.0, RACE / max(spec[-1], alone[-1]))
    tl = Timeline(LEAD + scale * max(spec[-1], alone[-1]) + HOLD)
    end = tl.period - 0.5
    body = [
        text(60, 70, "Speculative Decoding, One Verify Pass at a Time", 34, weight=700, font=SERIF),
        text(
            60,
            106,
            f"An EAGLE-3 head reads Qwen3-1.7B's hidden states and guesses up to {depth} tokens, as many as it's "
            "sure of. Qwen3-1.7B checks them in one pass and keeps them up to the first miss.",
            19,
            SOFT,
        ),
        text(60, 156, clip(f"Prompt: {t['prompt']}", 138), 18, SOFT, font=MONO),
    ]

    # Committed text first, then each pass's guesses on top of it.
    steps = [
        {**s, "drafted": [g.replace("\n", " ") for g in s["drafted"]], "target": s["target"].replace("\n", " ")}
        for s in steps
    ]
    written = [p for s in steps for p in [*s["drafted"][: s["kept"]], s["target"]]]
    spots = layout(written, MARGIN, 218, W - MARGIN)
    hidden: list[str] = []
    start, clock, settled, lowest = 0, LEAD, [], 218.0
    shown_at: dict[float, float] = {}  # each line of text, and the second something first lands on it
    for si, s in enumerate(steps):
        # Each pass plays in the time it took, with the page's rhythm of guess, check, then settle.
        t0, span = clock, scale * s["seconds"]
        clock += span
        verify, settle = t0 + 0.4 * span, t0 + 0.77 * span
        settled.append(settle)
        # Guesses run on from where the text ends, so each box fits its own guess.
        here = spots[start]
        for j, p in enumerate(layout(s["drafted"], MARGIN, here.y, W - MARGIN, at=here.x)):
            hidden.append(tl.show(f"g{si}_{j}", t0 + 0.05 * span * j, verify))
            shown_at.setdefault(p.y, t0 + 0.05 * span * j)
            lowest = max(lowest, p.y)
            body.append(chip(p, "guess", hidden[-1]))
            if j <= s["kept"]:
                hidden.append(tl.show(f"v{si}_{j}", verify, settle))
                body.append(chip(p, "kept" if j < s["kept"] else "miss", hidden[-1]))
        for j in range(s["kept"] + 1):
            spot = spots[start + j]
            hidden.append(tl.show(f"w{si}_{j}", settle, end, out=OUT))
            shown_at.setdefault(spot.y, settle)
            own = j == s["kept"]
            # The rule goes under the word and not the space before it.
            indent = spot.width * (1 - len(spot.text.lstrip()) / max(len(spot.text), 1))
            mark = (
                line(spot.x + indent, spot.y + 7, spot.x + spot.width, spot.y + 7, INK, 3)
                if own and spot.text.strip()
                else ""
            )
            word = text(spot.x, spot.y, spot.text, 20, INK, font=MONO, weight=700 if own else 400, fit=spot.width)
            body.append(f'<g class="{hidden[-1]}">{word}{mark}</g>')
        start += s["kept"] + 1

    # The alone run writes underneath on the same clock. The trace only keeps the guessed run's tokens,
    # which are the alone run's too when the texts match.
    bottom = max(lowest, *(p.y for p in spots))
    split, ry, small_at = len(body), bottom + 70, {}
    if timing["same"]:
        ay = bottom + 44
        body += [
            line(60, ay, W - 60, ay, FAINT, 1),
            text(60, ay + 30, " alone, on the same clock", 15, SOFT, lead="Qwen3-1.7B"),
        ]
        small = layout(written, MARGIN, ay + 60, W - MARGIN, char=SMALL, line_h=SMALL_H)
        for i, p in enumerate(small):
            hidden.append(tl.show(f"s{i}", LEAD + scale * alone[i], end, out=OUT))
            small_at.setdefault(p.y, LEAD + scale * alone[i])
            body.append(text(p.x, p.y, p.text, 15, INK, font=MONO, cls=hidden[-1], fit=p.width))
        ry = max(small_at) + 52
    under = len(body)

    span, n = W - 120, len(timing["alone"])
    for row, (label, ends, sizes) in enumerate(
        [
            (" with guesses", spec, [s["kept"] + 1 for s in steps]),
            (" alone", alone, [1] * n),
        ]
    ):
        y = ry + row * 50
        body += [text(60, y, label, 17, SOFT, lead="Qwen3-1.7B"), rect(60, y + 12, span, 10, FAINT)]
        x = 60.0
        for i, (finish, size) in enumerate(zip(ends, sizes, strict=True)):
            hidden.append(tl.show(f"r{row}_{i}", LEAD + scale * finish, end, out=OUT))
            # A pixel of overlap hides the antialiased seam between neighbouring segments.
            body.append(rect(x, y + 12, min(span * size / n + 1, 60 + span - x), 10, INK, cls=hidden[-1]))
            x += span * size / n
        hidden.append(tl.show(f"r{row}", LEAD + scale * ends[-1], end, out=OUT))
        body.append(text(W - 60, y, f"{ends[-1]:.2f} s", 17, INK, "end", weight=700, cls=hidden[-1]))

    by = ry + 130
    bw = (W - 120) / len(steps)
    body.append(text(60, by, "tokens gained per pass of the big model", 15, SOFT))
    tall = max(s["kept"] for s in steps)
    for si, s in enumerate(steps):
        x = 60 + si * bw
        body.append(rect(x, by + 14, bw - 8, (tall + 1) * 18, FAINT))
        cells = [
            rect(x, by + 14 + (tall - j) * 18, bw - 8, 16, INK)
            if j < s["kept"]
            else rect(x + 1, by + 15 + (tall - j) * 18, bw - 10, 14, SHEET, stroke=INK, width=2)
            for j in range(s["kept"] + 1)
        ]
        hidden.append(tl.show(f"b{si}", settled[si], end, out=OUT))
        body.append(f'<g class="{hidden[-1]}">{"".join(cells)}</g>')

    fy = by + 14 + (tall + 1) * 18 + 40
    kx = 60.0
    for draw, numeral, label in [
        (lambda x: rect(x, fy - 15, 22, 16, stroke=INK, dash="4 3"), 102, "guessed by the head"),
        (lambda x: rect(x, fy - 15, 22, 16, stroke=INK), 104, "kept by the big model"),
        (lambda x: rect(x, fy - 15, 22, 16, stroke=LOST, width=2, dash="4 3"), 106, "first miss, thrown away"),
        (lambda x: line(x, fy, x + 22, fy, INK, 3), 108, "the big model's own token"),
    ]:
        entry, width = callout(kx, fy, draw(kx), numeral, label)
        body.append(entry)
        kx += width + 34
    pace = "real speed" if scale == 1 else f"{1 / scale:.1f} times real speed"
    body.append(
        text(
            60,
            fy + 40,
            f"{'The same' if timing['same'] else 'Nearly the same'} text, {len(written)} tokens from {len(steps)} "
            f"big-model passes and {alone[-1] / spec[-1]:.2f} times as fast with guesses once the prompt is read. "
            "Every pass was timed on "
            f"{timing['device']}, and it plays at {pace}.",
            17,
            SOFT,
        )
    )
    # What sits under each text stays just below its last written line, the way the page grows.
    style = tl.css(",".join(f".{c}" for c in hidden) + "{opacity:0}", still=True)
    if small_at:
        body[under:] = [f'<g class="drop-small">{"".join(body[under:])}</g>']
        style += slide("drop-small", small_at, max(small_at), end, tl.period)
    body[split:] = [f'<g class="drop">{"".join(body[split:])}</g>']
    style += slide("drop", shown_at, bottom, end, tl.period)
    return document(
        "Speculative decoding: an EAGLE-3 head guesses tokens and a big model checks them in one pass",
        int(fy + 80),
        style,
        body,
    )


def slide(name: str, shown_at: dict[float, float], bottom: float, end: float, period: float) -> str:
    """CSS that lifts class `name` up to the first line, then drops it a line each time the text above wraps."""
    lines = sorted(shown_at.items(), key=lambda item: item[1])
    frames = [(0.0, lines[0][0])]
    for y, at in lines[1:]:
        frames += [(at, frames[-1][1]), (at + 0.25, y)]
    frames += [(end, frames[-1][1]), (end + 0.3, lines[0][0]), (period, lines[0][0])]
    keys = "".join(f"{100 * at / period:.2f}%{{transform:translateY({y - bottom:g}px)}}" for at, y in frames)
    return f"@keyframes {name}{{{keys}}}.{name}{{animation:{name} {period}s linear infinite}}"
