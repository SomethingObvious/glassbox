"""EAGLE-3 guessing for Qwen3-1.7B, and the big model writing alone, both timed pass by pass with a KV cache.

The head is one decoder layer. It reads three of the big model's hidden states plus the token that followed them and
guesses the token after that, then runs again on its own output for each further guess.
"""

import time
from collections.abc import Collection

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .draft import accepted

LAYERS = (2, 14, 25)  # hidden_states entries the head reads, the outputs of layers 1, 13 and 24 of 28
HEADS, KV_HEADS, DIM = 16, 8, 128


class Packed(torch.nn.Module):
    """A linear layer whose weight oneDNN lays out once, the way `torch.compile` freezes a model on CPU.

    Plain `F.linear` repacks the whole weight on every call with more than one row, so checking 5 tokens costs
    twice what writing one does. Packed, both stream the weight once.
    """

    def __init__(self, linear: torch.nn.Linear, rows: int) -> None:
        super().__init__()
        self.weight = torch.ops.mkldnn._reorder_linear_weight(linear.weight.detach(), rows)
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.mkldnn._linear_pointwise(x, self.weight, self.bias, "none", [], "")


def pack(model: torch.nn.Module, rows: int) -> None:
    """Swaps every linear layer for a packed one, for passes of about `rows` tokens."""
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(model, name, Packed(child, rows))
        else:
            pack(child, rows)


def rms(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * gain


class Head:
    """The guessing layer with its own KV cache. Its vocabulary is the big model's 32,000 most used tokens."""

    def __init__(self, weights: dict[str, torch.Tensor], embed: torch.nn.Embedding) -> None:
        self.dtype, self.dev = embed.weight.dtype, embed.weight.device
        floats = {k: v.to(self.dev, self.dtype) for k, v in weights.items() if v.is_floating_point()}
        self.w = {k: v for k, v in floats.items() if v.dim() == 1}  # the norms' gains
        self.lin: dict[str, torch.nn.Module] = {}
        for k, v in floats.items():
            if v.dim() == 2:
                lin = torch.nn.Linear(v.shape[1], v.shape[0], bias=False, device="meta")
                lin.weight = torch.nn.Parameter(v, requires_grad=False)
                self.lin[k.removesuffix(".weight")] = Packed(lin, 1) if self.dev.type == "cpu" else lin
        self.d2t = weights["d2t"].long().to(self.dev)  # head token id to the gap to the big model's id
        self.embed = embed
        self.inv = 1.0 / (1e6 ** (torch.arange(0, DIM, 2, device=self.dev).float() / DIM))
        self.k = self.v = torch.empty(KV_HEADS, 0, DIM, device=self.dev, dtype=self.dtype)

    @property
    def done(self) -> int:
        return self.k.shape[1]

    def keep(self, n: int) -> None:
        self.k, self.v = self.k[:, :n], self.v[:, :n]

    def rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        f = pos[:, None].float() * self.inv[None]
        cos, sin = (torch.cat([f, f], -1).cos().to(x.dtype), torch.cat([f, f], -1).sin().to(x.dtype))
        return x * cos + torch.cat([-x[..., DIM // 2 :], x[..., : DIM // 2]], -1) * sin

    def step(self, hidden: torch.Tensor, tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Adds one entry per token. `hidden` is the big model's fused states, or the head's own output."""
        w = self.w
        if hidden.shape[-1] != HEADS * DIM:
            hidden = self.lin["fc"](hidden.to(self.dtype))
        t, pos = len(tokens), torch.arange(self.done, self.done + len(tokens), device=self.dev)
        x = torch.cat(
            [
                rms(self.embed(torch.tensor(tokens, device=self.dev)), w["midlayer.input_layernorm.weight"]),
                rms(hidden, w["midlayer.hidden_norm.weight"]),
            ],
            -1,
        )
        q = self.rope(self.lin["midlayer.self_attn.q_proj"](x).view(t, HEADS, DIM).transpose(0, 1), pos)
        k = self.rope(self.lin["midlayer.self_attn.k_proj"](x).view(t, KV_HEADS, DIM).transpose(0, 1), pos)
        v = self.lin["midlayer.self_attn.v_proj"](x).view(t, KV_HEADS, DIM).transpose(0, 1)
        self.k, self.v = torch.cat([self.k, k], 1), torch.cat([self.v, v], 1)
        mask = torch.arange(self.done, device=self.dev)[None] <= pos[:, None]
        a = F.scaled_dot_product_attention(
            q, self.k.repeat_interleave(2, 0), self.v.repeat_interleave(2, 0), attn_mask=mask
        )
        h = hidden + self.lin["midlayer.self_attn.o_proj"](a.transpose(0, 1).reshape(t, HEADS * DIM))
        m = rms(h, w["midlayer.post_attention_layernorm.weight"])
        gate = F.silu(self.lin["midlayer.mlp.gate_proj"](m)) * self.lin["midlayer.mlp.up_proj"](m)
        h = h + self.lin["midlayer.mlp.down_proj"](gate)
        return h, self.lin["lm_head"](rms(h, w["norm.weight"]))

    def guess(self, logits: torch.Tensor) -> int:
        d = int(logits.argmax())
        return d + int(self.d2t[d])


class Big:
    """The big model with a KV cache that can be cut back after a miss."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model, self.cache, self.done = model, DynamicCache(), 0

    def feed(self, seq: list[int], states: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = torch.tensor([seq[self.done :]], device=self.model.device)
        out = self.model(x, past_key_values=self.cache, use_cache=True, output_hidden_states=states)
        self.done = len(seq)
        return out.logits[0], torch.cat([out.hidden_states[i][0] for i in LAYERS], -1) if states else None

    def keep(self, n: int) -> None:
        self.cache.crop(n)
        self.done = n


def clock(dev: torch.device) -> float:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def alone(model: torch.nn.Module, prompt: list[int], n: int) -> tuple[list[int], list[float]]:
    """The big model's next `n` greedy tokens, one pass each, and the seconds each pass took."""
    big, seq, times = Big(model), list(prompt), []
    for _ in range(n):
        start = clock(model.device)
        seq.append(int(big.feed(seq)[0][-1].argmax()))
        times.append(clock(model.device) - start)
    return seq[len(prompt) :], times


def guessed(
    model: torch.nn.Module,
    head: Head,
    prompt: list[int],
    depth: int,
    most: int,
    stop: Collection[int],
    sure: float = 0.0,
) -> list[tuple[list[int], int, int, float]]:
    """Each pass as (guesses, how many were kept, the big model's own token, seconds), until a stop token or `most`.

    The head guesses up to `depth` tokens a pass, and stops early once its own odds that every guess so far is right
    fall below `sure`. The first pass reads the prompt and has no guesses, as the head needs the big model's states.
    """
    head.keep(0)
    big, dev = Big(model), model.device
    start = clock(dev)
    logits, states = big.feed(prompt, states=True)
    seq = [*prompt, int(logits[-1].argmax())]
    passes = [([], 0, seq[-1], clock(dev) - start)]
    while seq[-1] not in stop and len(seq) - len(prompt) < most:
        start, base = clock(dev), len(seq)
        # Catch the head up on the big model's real states, then let it run on its own output.
        h, logits = head.step(states, seq[head.done + 1 : base])
        real = head.done
        guesses, odds = [head.guess(logits[-1])], logits[-1].float().softmax(-1).max().item()
        while len(guesses) < depth and odds >= sure:
            h, logits = head.step(h[-1:], guesses[-1:])
            guesses.append(head.guess(logits[-1]))
            odds *= logits[-1].float().softmax(-1).max().item()
        head.keep(real)
        logits, states = big.feed(seq + guesses, states=True)
        wants = logits[-(len(guesses) + 1) :].argmax(-1).tolist()
        kept = accepted(guesses, wants)
        # A kept guess can end the answer, and nothing guessed after it counts.
        if (end := next((j for j, g in enumerate(guesses[:kept]) if g in stop), None)) is not None:
            guesses = guesses[:end]
            kept = end
        seq += [*guesses[:kept], wants[kept]]
        big.keep(base + kept)
        states = states[: kept + 1]
        passes.append((guesses, kept, wants[kept], clock(dev) - start))
    return passes
