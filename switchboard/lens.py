"""The logit lens: Qwen3-0.6B's next-token guess read out after each of its layers."""

from typing import TypedDict

from . import pieces

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = (
    'In French, the word for dog is "',
    "The chemical symbol for gold is",
    "The Eiffel Tower is in the city of",
    "In Python, you print text with the function",
    "The first president of the United States was George",
)
TOP = 5


class Run(TypedDict):
    text: str
    tokens: list[str]
    # For each layer and position, the TOP likeliest next tokens as [text, probability] pairs.
    layers: list[list[list[tuple[str, float]]]]


class Trace(TypedDict):
    model: str
    runs: list[Run]


def trace() -> Trace:
    """Runs each prompt once on CPU and decodes every layer's output with the model's own final norm and head."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    seen: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: tuple, out: torch.Tensor | tuple) -> None:
        seen.append(out[0] if isinstance(out, tuple) else out)

    for layer in model.model.layers:
        layer.register_forward_hook(hook)
    runs: list[Run] = []
    with torch.inference_mode():
        for text in PROMPTS:
            seen.clear()
            ids = tok(text, return_tensors="pt").input_ids
            model(ids)
            layers = []
            for h in seen:
                probs = model.lm_head(model.model.norm(h))[0].softmax(-1)
                top = probs.topk(TOP, -1)
                layers.append(
                    [
                        [(pieces(tok, [i])[0], round(p, 4)) for i, p in zip(row_ids, row_p, strict=True)]
                        for row_ids, row_p in zip(top.indices.tolist(), top.values.tolist(), strict=True)
                    ]
                )
            runs.append({"text": text, "tokens": pieces(tok, ids[0].tolist()), "layers": layers})
            print(text, "->", [layer[-1][0] for layer in layers[-4:]])
    return {"model": MODEL, "runs": runs}
