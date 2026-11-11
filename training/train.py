"""Trains an EAGLE-3 head for Qwen3-1.7B on the GPU from the big model's own answers.

Arguments: start weights, train jsonl, test jsonl, output folder, epochs.
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from switchboard.eagle import LAYERS

START, TRAIN, TEST, OUT, EPOCHS = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
K, ACCUM, LR = 4, 8, 5e-5  # guesses unrolled, rows per optimizer step, peak learning rate
dev = "cuda"
torch.manual_seed(0)
random.seed(0)

big = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(dev).eval()
for p in big.parameters():
    p.requires_grad_(False)
embed = big.model.embed_tokens

raw = torch.load(START, map_location="cpu", weights_only=True)
d2t, t2d = raw["d2t"].long().to(dev), raw["t2d"].bool().to(dev)
vocab = torch.arange(t2d.numel(), device=dev)[t2d]
assert torch.equal(vocab, torch.arange(len(d2t), device=dev) + d2t), "d2t and t2d disagree"
W = torch.nn.ParameterDict(
    {k.replace(".", "__"): torch.nn.Parameter(v.float().to(dev)) for k, v in raw.items() if v.is_floating_point()}
)
inv = 1.0 / (1e6 ** (torch.arange(0, 128, 2, device=dev).float() / 128))


def w(name: str) -> torch.Tensor:
    return W[name.replace(".", "__")]


def rms(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    f = pos[:, None].float() * inv[None]
    cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()
    rot = torch.cat([-x[..., 64:], x[..., :64]], -1)
    return x * cos + rot * sin


def unroll(feats: torch.Tensor, tokens: torch.Tensor) -> list[torch.Tensor]:
    """Draft logits for K guesses at every entry. Entry t pairs feature t with token t + 1 and guesses token t + 2.

    Guess j at entry t runs on the head's own output from guess j - 1 at entry t - 1, and attends to the real entries up
    to t - j + 1 plus its own chain, which is what it sees while decoding.
    """
    e = len(tokens) - 1
    pos = torch.arange(e, device=dev)
    emb = rms(embed(tokens[1:]).float(), w("midlayer.input_layernorm.weight"))
    hidden = feats[:e] @ w("fc.weight").T
    keys, values, outs = [], [], []
    t = pos[:, None]
    for j in range(1, K + 1):
        x = torch.cat([emb, rms(hidden, w("midlayer.hidden_norm.weight"))], -1)
        q = rope((x @ w("midlayer.self_attn.q_proj.weight").T).view(e, 16, 128).transpose(0, 1), pos)
        keys.append(rope((x @ w("midlayer.self_attn.k_proj.weight").T).view(e, 8, 128).transpose(0, 1), pos))
        values.append((x @ w("midlayer.self_attn.v_proj.weight").T).view(e, 8, 128).transpose(0, 1))
        masks = [pos[None] <= t - j + 1] + [pos[None] == t - j + i for i in range(2, j + 1)]
        mask = torch.cat(masks, 1)
        mask[:, 0] |= ~mask.any(1)  # rows before a guess can exist attend somewhere, and are masked from the loss
        kk = torch.cat(keys, 1).repeat_interleave(2, 0)
        vv = torch.cat(values, 1).repeat_interleave(2, 0)
        a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
        h = hidden + a.transpose(0, 1).reshape(e, 2048) @ w("midlayer.self_attn.o_proj.weight").T
        m = rms(h, w("midlayer.post_attention_layernorm.weight"))
        gate = F.silu(m @ w("midlayer.mlp.gate_proj.weight").T) * (m @ w("midlayer.mlp.up_proj.weight").T)
        h = h + gate @ w("midlayer.mlp.down_proj.weight").T
        outs.append(rms(h, w("norm.weight")) @ w("lm_head.weight").T)
        hidden = torch.cat([torch.zeros_like(h[:1]), h[:-1]])
    return outs


def step(row: dict, train: bool) -> tuple[torch.Tensor, list[float], int]:
    tokens = torch.tensor(row["prompt"] + row["answer"], device=dev)
    with torch.no_grad():
        out = big(tokens[None], output_hidden_states=True)
        feats = torch.cat([out.hidden_states[i][0] for i in LAYERS], -1).float()
        want = F.softmax(out.logits[0, 1:, vocab].float(), -1)  # row t: the big model after token t + 1
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=train):
        outs = unroll(feats, tokens)
    e = len(tokens) - 1
    start = len(row["prompt"]) - 2  # first entry whose guess is an answer token
    loss, hits = 0.0, []
    for j, logits in enumerate(outs, 1):
        rows = torch.arange(e, device=dev)
        keep = (rows >= max(start, j - 1)).float()
        ce = -(want * F.log_softmax(logits.float(), -1)).sum(-1)
        loss = loss + (ce * keep).sum() / keep.sum().clamp(min=1)
        right = (logits.argmax(-1) == want.argmax(-1)).float()
        hits.append(float((right * keep).sum() / keep.sum().clamp(min=1)))
    return loss / K, hits, e


def evaluate(rows: list[dict]) -> list[float]:
    total = [0.0] * K
    with torch.no_grad():
        for row in rows:
            _, hits, _ = step(row, False)
            total = [a + b for a, b in zip(total, hits, strict=True)]
    return [t / len(rows) for t in total]


def save(tag: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state = {k.replace("__", "."): v.detach().to(torch.float16).cpu().contiguous() for k, v in W.items()}
    state["d2t"], state["t2d"] = d2t.cpu(), t2d.cpu()
    save_file(state, str(OUT / f"{tag}.safetensors"))


train = [json.loads(line) for line in Path(TRAIN).read_text(encoding="utf-8").splitlines()]
test = [json.loads(line) for line in Path(TEST).read_text(encoding="utf-8").splitlines()]
opt = torch.optim.AdamW(W.values(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
total = EPOCHS * len(train) // ACCUM
warm = 100


def lr_at(s: int) -> float:
    if s < warm:
        return LR * (s + 1) / warm
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


print("start", [f"{a:.1%}" for a in evaluate(test)], flush=True)
started, s, seen = time.perf_counter(), 0, 0
for epoch in range(EPOCHS):
    random.shuffle(train)
    for row in train:
        loss, _, _ = step(row, True)
        (loss / ACCUM).backward()
        seen += 1
        if seen % ACCUM == 0:
            for g in opt.param_groups:
                g["lr"] = lr_at(s)
            torch.nn.utils.clip_grad_norm_(W.values(), 0.5)
            opt.step()
            opt.zero_grad(set_to_none=True)
            s += 1
            if s % 200 == 0:
                acc = evaluate(test)
                rate = seen / (time.perf_counter() - started)
                print(
                    f"epoch {epoch} step {s}/{total} loss {float(loss):.3f} test {[f'{a:.1%}' for a in acc]} "
                    f"{rate:.1f} rows a second",
                    flush=True,
                )
    save(f"epoch{epoch}")
print("done in", round(time.perf_counter() - started), "s", flush=True)
