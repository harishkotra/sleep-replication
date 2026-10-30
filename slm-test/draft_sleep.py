"""Stage 1 of the sleep arm: consolidate + dream with the teacher.

Runs wherever the Hermes endpoint is configured (this machine). Writes cached
JSON so the GPU box never needs credentials.

  data/facts.json          consolidated atomic facts, one per session fact
  data/dreams_drafts.json  candidate two-hop questions, unfiltered

The learning-signal filter is applied later, on the box that trains, because it
must be the STUDENT model that fails the question -- that is the whole point.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from teacher import DREAM_SYS, SEED_SYS, json_reply

HERE = Path(__file__).parent
D = HERE / "data"
RNG = random.Random(11)
N_CAND = 96


def main() -> None:
    sessions = json.loads((D / "sessions.json").read_text())

    facts = []
    for s in sessions:
        out = json_reply(SEED_SYS, s["text"]) or {}
        got = 0
        for it in out.get("facts", []):
            t = str(it.get("text", "")).strip()
            if t:
                facts.append({"text": t, "session": s["session"], "topic": it.get("topic", "")})
                got += 1
        print(f"session {s['session']} ({s['subsystem']}): {got} facts")
    (D / "facts.json").write_text(json.dumps(facts, indent=1))
    words = sum(len(s["text"].split()) for s in sessions)
    print(f"{len(facts)} facts from {words} raw words -> {(D / 'facts.json').name}")

    pairs = []
    while len(pairs) < N_CAND:
        a, b = RNG.sample(facts, 2)
        if a["session"] != b["session"]:
            pairs.append((a, b))

    drafts = []
    for i, (a, b) in enumerate(pairs, 1):
        out = json_reply(DREAM_SYS, f"Memory A: {a['text']}\nMemory B: {b['text']}\n") or {}
        if out.get("question") and out.get("answer"):
            drafts.append({"question": out["question"], "answer": out["answer"],
                           "a": a["text"], "b": b["text"]})
        if i % 24 == 0:
            print(f"  {i}/{len(pairs)} pairs -> {len(drafts)} usable dreams")
    (D / "dreams_drafts.json").write_text(json.dumps(drafts, indent=1))
    print(f"wrote {len(drafts)} dream drafts -> {(D / 'dreams_drafts.json').name}")


if __name__ == "__main__":
    main()
