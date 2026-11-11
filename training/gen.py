"""Qwen3-1.7B answers every prompt greedily on the GPU. Arguments: prompts json, output jsonl, batch size."""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from switchboard import draft

src, dst, batch = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok = AutoTokenizer.from_pretrained(draft.TARGET, padding_side="left")
big = AutoModelForCausalLM.from_pretrained(draft.TARGET, dtype=torch.bfloat16).to("cuda").eval()
items = json.loads(Path(src).read_text(encoding="utf-8"))


def chat(item: dict) -> list[int]:
    text = draft.ask(item["text"]) if item["kind"] == "hn" else item["text"]
    s = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tok(s).input_ids


rows = sorted(((chat(it), it["kind"]) for it in items), key=lambda r: len(r[0]))
started, done = time.perf_counter(), 0
with open(dst, "w", encoding="utf-8") as out, torch.inference_mode():
    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        longest = max(len(p) for p, _ in part)
        ids = torch.full((len(part), longest), tok.pad_token_id)
        mask = torch.zeros_like(ids)
        for j, (p, _) in enumerate(part):
            ids[j, longest - len(p) :] = torch.tensor(p)
            mask[j, longest - len(p) :] = 1
        cap = 160 if all(k == "hn" for _, k in part) else 256
        gen = big.generate(
            ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=cap, do_sample=False, pad_token_id=tok.pad_token_id
        )[:, longest:].tolist()
        for (p, kind), g in zip(part, gen, strict=True):
            while g and g[-1] == tok.pad_token_id:
                g.pop()
            out.write(json.dumps({"kind": kind, "prompt": p, "answer": g}) + "\n")
        done += len(part)
        if (i // batch) % 10 == 0:
            rate = done / (time.perf_counter() - started)
            print(f"{done}/{len(rows)} answered, {rate:.1f} a second", flush=True)
print("done in", round(time.perf_counter() - started), "s")
