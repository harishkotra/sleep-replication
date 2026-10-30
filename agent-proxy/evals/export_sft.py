"""Export consolidated memory as SFT JSONL — the optional step from store to weights.

Run the sleep cycles first, then:
    python evals/export_sft.py evals/work/agent/long_term.json out/sft.jsonl

The JSONL is chat-format and drops straight into TRL SFTTrainer or
Unsloth's `train_on_responses_only`, i.e. the same consolidated memories can be
baked into parameters (LoRA) instead of retrieved. Dream samples are included
as high-value rehearsal pairs with their answers as the target.
"""
from __future__ import annotations

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "evals/work/agent/long_term.json"
out = sys.argv[2] if len(sys.argv) > 2 else "out/sft.jsonl"

nodes = json.load(open(path))["nodes"]
rows: list[dict] = []

for n in nodes:
    if n["kind"] == "dream" and "->" in n["text"]:
        q, a = n["text"].split("->", 1)
        q = q.replace("Rehearsed:", "").strip()
        rows.append({"messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": a.strip()}]})
    elif n["kind"] in ("fact", "skill"):
        rows.append({"messages": [{"role": "user", "content": "What do you know about this?"},
                                  {"role": "user", "content": n["text"]},
                                  {"role": "assistant", "content": n["text"]}]})

with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} SFT rows -> {out}")
