"""Retention experiment: does sleep beat plain in-context memory / chunk RAG?

Conditions
  no_memory      frozen model, no memory at all
  transcript_rag raw session transcripts retrieved (classic ICL agent)
  chunk_rag      standard RAG over fact-sized chunks
  sleep          consolidated facts + rehearsed dreams, raw transcripts pruned

Writes evals/results.json + prints a markdown table.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sleep_agent import SleepAgent, get_llm, ask  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = json.load(open(os.path.join(ROOT, "data", "probe_corpus.json")))
WORK = os.path.join(ROOT, "evals", "work")
LLM = get_llm()
JUDGE_SYS = ("You grade a factual answer against a reference. Ignore phrasing, judge meaning only. "
             "Reply with exactly CORRECT or INCORRECT.")
ANS_SYS = """You are an assistant with a long-term memory store.
Answer the question using ONLY the memory below. Memory overrides your prior beliefs.
If the memory lacks the answer, say "I don't know". Be terse.

MEMORY:
{memory}"""


def judge(question: str, gold: str, got: str) -> bool:
    v = ask(LLM, JUDGE_SYS, f"Q: {question}\nReference: {gold}\nModel answer: {got}", max_tokens=900)
    v = v.strip().upper()
    return "CORRECT" in v and "INCORRECT" not in v


def chunks(text: str) -> list[str]:
    return [ln.strip("[]") for ln in text.split("\n") if len(ln.split()) > 5]


def top_k_tfidf(query: str, pool: list[str], k: int) -> list[str]:
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(pool)
    sims = cosine_similarity(v.transform([query]), v.transform(pool))[0]
    return [pool[i] for i in np.argsort(-sims)[:k]]


def answer(memory: str, q: str) -> str:
    return ask(LLM, ANS_SYS.format(memory=memory or "(none)"), q, max_tokens=800)


def build_sleep_agent() -> tuple[SleepAgent, list[dict], int]:
    home = os.path.join(WORK, "agent")
    shutil.rmtree(home, ignore_errors=True)
    agent = SleepAgent(home, k=6, dreams_per_sleep=6)
    reports, ctx_tokens = [], 0
    for s in DATA["sessions"]:
        agent.remember(s["transcript"])
        reports.append(agent.sleep(n_dreams=6, dream_seed=s["id"]))
        ctx_tokens += agent.short.token_count()          # after pruning
    return agent, reports, ctx_tokens


def main() -> None:
    os.makedirs(WORK, exist_ok=True)
    qs = DATA["questions"]
    transcripts = [s["transcript"] for s in DATA["sessions"]]
    chunk_pool = [c for t in transcripts for c in chunks(t)]

    agent, sleep_reports, sleep_ctx = build_sleep_agent()
    print(json.dumps([{k: r[k] for k in ("session", "seeded", "dreams", "pruned", "seconds")}
                      for r in sleep_reports], indent=1))
    print("sleep store:", agent.long.stats())

    conds = {
        "no_memory": lambda q: answer("", q),
        "transcript_rag": lambda q: answer("\n\n".join(top_k_tfidf(q, transcripts, 2)), q),
        "chunk_rag": lambda q: answer("\n".join(top_k_tfidf(q, chunk_pool, 6)), q),
        "sleep": lambda q: agent.ask(q),
    }
    ctx_size = {"no_memory": 0,
                "transcript_rag": int(np.mean([len(" ".join(top_k_tfidf(q["q"], transcripts, 2)).split()) for q in qs])),
                "chunk_rag": int(np.mean([len(" ".join(top_k_tfidf(q["q"], chunk_pool, 6)).split()) for q in qs])),
                "sleep": int(np.mean([len(agent.long.context_block(q["q"], k=6).split()) for q in qs]))}

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for name, fn in conds.items():
            answers = list(ex.map(lambda q: fn(q["q"]), qs))
            verdicts = list(ex.map(lambda p: judge(p[0]["q"], p[0]["a"], p[1]), zip(qs, answers)))
            per_type = {}
            for q, ok in zip(qs, verdicts):
                per_type.setdefault(q["type"], []).append(ok)
            results[name] = {
                "accuracy": round(sum(verdicts) / len(verdicts), 3),
                "by_type": {t: round(sum(v) / len(v), 3) for t, v in per_type.items()},
                "context_tokens_per_query": ctx_size[name],
                "samples": [{"q": q["q"], "gold": q["a"], "got": a, "ok": bool(o)}
                            for q, a, o in zip(qs, answers, verdicts)],
            }
            print(f"{name}: acc={results[name]['accuracy']} by_type={results[name]['by_type']}")

    out = {"corpus_note": DATA["note"], "n_questions": len(qs),
           "sleep_cycles": [{k: r[k] for k in ("session", "seeded", "dreams", "pruned")} for r in sleep_reports],
           "final_store": agent.long.stats(), "results": results}
    json.dump(out, open(os.path.join(ROOT, "evals", "results.json"), "w"), indent=1)

    print("\n| condition | accuracy | direct | derived | ctx tokens/query |\n|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['accuracy']:.0%} | {r['by_type'].get('direct',0):.0%} | "
              f"{r['by_type'].get('derived',0):.0%} | {r['context_tokens_per_query']} |")


if __name__ == "__main__":
    main()
