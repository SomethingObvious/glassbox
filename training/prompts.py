"""Builds training and held-out prompts: past HN titles and oasst1 first turns. Argument: output folder."""

import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
NOW = int(time.time())
HOLD = NOW - 3 * 86400  # titles from the last three days are only for testing


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "switchboard-training"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


titles: dict[str, int] = {}
before = NOW
while len(titles) < 6000:
    page = get(
        "https://hn.algolia.com/api/v1/search_by_date?tags=story&hitsPerPage=1000"
        f"&numericFilters=points>40,created_at_i<{before}"
    )
    hits = page["hits"]
    if not hits:
        break
    for h in hits:
        if h.get("title"):
            titles[h["title"].strip()] = h["created_at_i"]
    before = min(h["created_at_i"] for h in hits)
    print(len(titles), "titles back to", time.strftime("%Y-%m-%d", time.gmtime(before)))

files = get("https://huggingface.co/api/datasets/OpenAssistant/oasst1/tree/main/data")
name = next(f["path"] for f in files if f["path"].endswith(".parquet") and "train" in f["path"])
local = OUT / "oasst1-train.parquet"
urllib.request.urlretrieve(f"https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/{name}", local)
table = pq.read_table(local, columns=["text", "role", "lang", "parent_id"]).to_pylist()
chats = sorted(
    {r["text"].strip() for r in table if r["role"] == "prompter" and r["lang"] == "en" and r["parent_id"] is None}
)
chats = [c for c in chats if 10 <= len(c) <= 600]

random.seed(0)
random.shuffle(chats)
recent = [t for t, at in titles.items() if at >= HOLD]
old = [t for t, at in titles.items() if at < HOLD]
random.shuffle(old)
train = [{"kind": "hn", "text": t} for t in old] + [{"kind": "chat", "text": c} for c in chats[:2500]]
test = [{"kind": "hn", "text": t} for t in recent[:40]] + [{"kind": "chat", "text": c} for c in chats[2500:2520]]
random.shuffle(train)
(OUT / "train.json").write_text(json.dumps(train), encoding="utf-8")
(OUT / "test.json").write_text(json.dumps(test), encoding="utf-8")
print(f"train {len(train)} ({len(old)} titles, {min(2500, len(chats))} chats), test {len(test)}")
