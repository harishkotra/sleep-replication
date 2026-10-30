"""Interference experiment: 5 sequential sessions that revise earlier facts.

compares chunk_rag (all raw transcripts, top-6) vs sleep (store retires superseded facts).
Writes evals/results_updates.json.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sleep_agent import SleepAgent, ask
from sleep_agent.llm import get_llm
from evals.eval_retention import ANS_SYS, JUDGE_SYS, chunks, judge  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = json.load(open(os.path.join(ROOT, "data", "update_stream.json")))
WORK = os.path.join(ROOT, "evals", "work")
LLM = get_llm()


def top_k(query: str, pool: list[str], k: int) -> list[str]:
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(pool)
    sims = cosine_similarity(v.transform([query]), v.transform(pool))[0]
    return [pool[i] for i in np.argsort(-sims)[:k]]


def main() -> None:
    os.makedirs(WORK, exist_ok=True)
    home = os.path.join(WORK, "updates")
    shutil.rmtree(home, ignore_errors=True)
    agent = SleepAgent(home, k=6, dreams_per_sleep=0)
    for s in DATA["sessions"]:
        agent.remember(s["transcript"])
        rep = agent.sleep(n_dreams=0, dream_seed=s["id"])
        print(f"session {s['id']}: +{rep['seeded']['facts']}f/{rep['seeded']['skills']}s "
              f"superseded={rep['seeded']['superseded']}")

    transcripts = [s["transcript"] for s in DATA["sessions"]]
    pool = [c for t in transcripts for c in chunks(t)]
    print("store:", agent.long.stats())

    out = {}
    for name in ("chunk_rag", "sleep"):
        rows = []
        for q in DATA["questions"]:
            if name == "sleep":
                got = agent.ask(q["q"])
            else:
                got = ask(LLM, ANS_SYS.format(memory="\n".join(top_k(q["q"], pool, 6))), q["q"], max_tokens=800)
            ok = judge(q["q"], q["a"], got)
            rows.append({"q": q["q"], "gold": q["a"], "got": got, "ok": ok})
        out[name] = {"accuracy": round(sum(r["ok"] for r in rows) / len(rows), 3), "samples": rows}
        print(name, out[name]["accuracy"])

    out["store"] = agent.long.stats()
    json.dump(out, open(os.path.join(ROOT, "evals", "results_updates.json"), "w"), indent=1)
    print("\n| condition | current-value accuracy |")
    for n in ("chunk_rag", "sleep"):
        print(f"| {n} | {out[n]['accuracy']:.0%} |")


if __name__ == "__main__":
    main()
