"""What rounding Qwen3-0.6B's weights to fewer bits does, from one row of weights up to the text it writes."""

from typing import TYPE_CHECKING, NamedTuple, TypedDict

from . import pieces
from .draft import PROMPT

if TYPE_CHECKING:
    import torch

MODEL = "Qwen/Qwen3-0.6B"
NEW = 64  # tokens each format writes


class Format(NamedTuple):
    name: str
    bits: int  # 0 keeps the weights as they are
    group: int  # weights sharing one scale, 0 for a whole row
    fp8: bool = False


FORMATS = (
    Format("bf16", 0, 0),
    Format("int8 per row", 8, 0),
    Format("fp8 e4m3 per row", 8, 0, fp8=True),
    Format("int4 in groups of 128", 4, 128),
    Format("int4 per row", 4, 0),
    Format("int3 in groups of 128", 3, 128),
)


class Row(TypedDict):
    where: str  # which matrix and row
    values: list[float]


class Run(TypedDict):
    format: str
    bits: int
    group: int
    fp8: bool
    tokens: list[str]
    same: int  # tokens before the first one that differs from bf16
    agree: float  # share of bf16's next-token picks this format also makes on bf16's text
    kl: float  # mean KL divergence from bf16's next-token distribution, in nats


class Trace(TypedDict):
    model: str
    prompt: str
    rows: list[Row]
    runs: list[Run]


def rounded(w: "torch.Tensor", f: Format) -> "torch.Tensor":
    """Rounds each group of a weight matrix to `f`, min to max over 2^bits levels, and scales it back."""
    import torch

    if not f.bits:
        return w
    g = w.reshape(-1, f.group or w.shape[-1])
    if f.fp8:
        scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        return ((g / scale).to(torch.float8_e4m3fn).to(w.dtype) * scale).reshape(w.shape)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    step = ((hi - lo) / (2**f.bits - 1)).clamp_min(1e-12)
    return (((g - lo) / step).round() * step + lo).reshape(w.shape)


def trace() -> Trace:
    """Writes NEW tokens greedily in every format on CPU, then scores each on bf16's text."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    linears = {n: m for n, m in model.model.layers.named_modules() if isinstance(m, torch.nn.Linear)}
    original = {n: m.weight.detach().clone() for n, m in linears.items()}

    # The row whose biggest weight stands furthest out from the rest, and a row that is middling by the same measure.
    spread = []
    for n, w in original.items():
        ratio = w.abs().amax(-1) / w.std(-1)
        spread += [(float(r), n, i) for i, r in enumerate(ratio.tolist())]
    spread.sort()
    rows: list[Row] = [
        {"where": f"layer {n.split('.')[0]} {n.split('.', 1)[1]}, row {i}", "values": original[n][i].tolist()}
        for _, n, i in (spread[-1], spread[len(spread) // 2])
    ]
    for row in rows:
        row["values"] = [float(f"{v:.5g}") for v in row["values"]]

    chat = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt = tok(chat, return_tensors="pt").input_ids
    start = prompt.shape[1]

    def write(f: Format) -> list[int]:
        for n, m in linears.items():
            m.weight.copy_(rounded(original[n], f))
        return model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=NEW, do_sample=False)[
            0, start:
        ].tolist()

    def scores(new: list[int]) -> torch.Tensor:
        ids = torch.cat([prompt[0], torch.tensor(new)])
        return model(ids[None]).logits[0, start - 1 : -1].log_softmax(-1)

    runs: list[Run] = []
    with torch.inference_mode():
        ref = write(FORMATS[0])
        base = scores(ref)
        for f in FORMATS:
            new = write(f)
            logp = scores(ref)
            same = next((i for i, (a, b) in enumerate(zip(new, ref, strict=False)) if a != b), min(len(new), len(ref)))
            runs.append(
                {
                    "format": f.name,
                    "bits": f.bits or 16,
                    "group": f.group,
                    "fp8": f.fp8,
                    "tokens": pieces(tok, new),
                    "same": same,
                    "agree": round(float((logp.argmax(-1) == base.argmax(-1)).float().mean()), 4),
                    "kl": round(float((base.exp() * (base - logp)).sum(-1).mean()), 5),
                }
            )
            print(f.name, same, runs[-1]["agree"], runs[-1]["kl"])
    return {"model": MODEL, "prompt": PROMPT, "rows": rows, "runs": runs}
