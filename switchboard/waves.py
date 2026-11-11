"""One GEMM's output tiles going out to an H100's SMs in waves, and the idle tail of the last wave."""

import math
from dataclasses import dataclass

from .svg import FAINT, INK, LOST, MONO, SERIF, SHEET, SOFT, Timeline, W, callout, document, rect, text


@dataclass(frozen=True)
class Plan:
    m: int
    n: int
    tile: int
    sms: int

    @property
    def rows(self) -> int:
        return math.ceil(self.m / self.tile)

    @property
    def cols(self) -> int:
        return math.ceil(self.n / self.tile)

    @property
    def tiles(self) -> int:
        return self.rows * self.cols

    @property
    def waves(self) -> int:
        return math.ceil(self.tiles / self.sms)

    def busy(self, wave: int) -> int:
        return min(self.sms, self.tiles - wave * self.sms)

    @property
    def efficiency(self) -> float:
        """Share of SM time doing work, when every tile takes the same time."""
        return self.tiles / (self.waves * self.sms)


# Llama 3 8B's MLP up projection over 512 tokens, in BF16 on an H100 SXM.
PLAN, K = Plan(m=512, n=14336, tile=128, sms=132), 4096
PERIOD, START, STEP, FADE = 15.0, 1.2, 2.7, 0.5


def render(p: Plan = PLAN) -> str:
    tl, end = Timeline(PERIOD), PERIOD - 0.6
    body = [
        text(60, 70, "One Matrix Multiply on an H100", 34, weight=700, font=SERIF),
        text(
            60,
            106,
            f"C = A × B with M = {p.m:,}, N = {p.n:,} and K = {K:,} in BF16. Its {p.tiles} output tiles of "
            f"{p.tile} × {p.tile} go out to {p.sms} SMs, one tile per SM at a time.",
            19,
            SOFT,
        ),
    ]
    # Everything eases in and out, since whole rows of light blocks blinking at once is hard on the eyes.
    running, done, label, bar = [], [], [], []
    for w in range(p.waves):
        on, off = START + w * STEP, START + (w + 1) * STEP
        final = w == p.waves - 1
        running.append(tl.show(f"a{w}", on, end if final else off + FADE / 2, fade=FADE))
        done.append(tl.show(f"c{w}", off - FADE / 2, end, lit=0.3, fade=FADE))
        label.append(tl.show(f"l{w}", on, end if final else off, fade=0.3))
        bar.append(tl.show(f"b{w}", on, end, fade=FADE))
    last = START + (p.waves - 1) * STEP
    # An SM stays lit for as long as it has tiles, so only the ones the last wave leaves out go dark.
    working = tl.show("work", START, end, fade=2 * FADE)
    resting = tl.show("rest", START, last + FADE / 2, fade=2 * FADE)
    idle = tl.show("idle", last, end, fade=FADE)

    # A tile stays as a light wash once its wave is done.
    x0, y0, cw = 60, 154, (W - 120) / p.cols
    body.append(text(x0, y0 - 12, f"output C, {p.rows} × {p.cols} tiles", 15, SOFT))
    for i in range(p.tiles):
        r, c = divmod(i, p.cols)
        x, y, w = x0 + c * cw, y0 + r * cw, i // p.sms
        body += [
            rect(x, y, cw - 2, cw - 2, FAINT),
            rect(x, y, cw - 2, cw - 2, INK, done[w]),
            rect(x, y, cw - 2, cw - 2, INK, running[w]),
        ]

    gx, gy, per_row, gap = 60, 254, 22, 8
    side = (W - 120 - (per_row - 1) * gap) / per_row
    body.append(text(gx, gy - 12, f"{p.sms} streaming multiprocessors", 15, SOFT))
    for s in range(p.sms):
        r, c = divmod(s, per_row)
        x, y = gx + c * (side + gap), gy + r * (side + gap)
        body.append(rect(x, y, side, side, stroke=FAINT))
        body.append(rect(x, y, side, side, INK, working if s < p.busy(p.waves - 1) else resting))
        for w in range(p.waves):
            if s < p.busy(w):
                tile = f"t{w * p.sms + s}"
                body.append(text(x + side / 2, y + side / 2 + 6, tile, 15, SHEET, "middle", 700, MONO, label[w]))
        if s >= p.busy(p.waves - 1):
            body.append(rect(x + 2, y + 2, side - 4, side - 4, cls=idle, stroke=LOST, width=2, dash="6 5"))

    ly = gy + math.ceil(p.sms / per_row) * (side + gap) + 34
    for w in range(p.waves):
        n = p.busy(w)
        note = f"wave {w + 1} of {p.waves}: {n} of {p.sms} SMs busy"
        note += f", {p.sms - n} sit idle until it ends" if n < p.sms else ""
        body.append(text(gx, ly, note, 22, LOST if n < p.sms else INK, weight=700, cls=label[w]))
    ty, bw = ly + 34, (W - 120 - (p.waves - 1) * 10) / p.waves
    for w in range(p.waves):
        x, share = gx + w * (bw + 10), p.busy(w) / p.sms
        body.append(rect(x, ty, bw, 26, FAINT))
        body.append(
            f'<g class="{bar[w]}">{rect(x, ty, bw * share, 26, INK)}'
            f"{text(x + 10, ty + 19, f'{100 * share:.0f}% busy', 15, SHEET, weight=700, font=MONO)}</g>"
        )
    fy = ty + 66
    kx = gx
    for draw, numeral, label_text in [
        (lambda x: rect(x, fy - 15, 22, 16, INK), 302, "an SM running a tile"),
        (lambda x: rect(x, fy - 15, 22, 16, stroke=LOST, width=2, dash="4 3"), 304, "an SM idle in this wave"),
    ]:
        entry, width = callout(kx, fy, draw(kx), numeral, label_text)
        body.append(entry)
        kx += width + 34
    body.append(
        text(
            W - 60,
            fy,
            f"{p.tiles / p.sms:.2f} waves of work take {p.waves} waves of time, so the GPU runs this multiply at "
            f"{100 * p.efficiency:.0f}% of what its SMs could do",
            17,
            INK,
            "end",
            weight=700,
        )
    )
    hidden = ",".join(f".{c}" for c in [*running, *done, *label, *bar, working, resting, idle])
    return document(
        "A matrix multiply's tiles going out to an H100's SMs in waves",
        int(fy + 40),
        tl.css(hidden + "{opacity:0}", still=True),
        body,
    )
