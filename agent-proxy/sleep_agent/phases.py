"""The two sleep stages + pruning, mapped onto the paper.

1. Memory consolidation / Knowledge Seeding
   Fragile in-context traces -> atomic, self-contained durable facts.
   (Paper: on-policy distillation + RL imitation learning into expanded
   parameters. Here the "student" is the durable store, and the distillation
   objective is: rewrite so the fact is answerable with zero context.)

2. Dreaming
   Random recombination of unrelated memories ("random expert activation")
   to synthesise questions that no single memory can answer, then a
   learning-signal filter ("gradient-based data selection"): keep only dreams
   the frozen model would get WRONG on its own, because those are the samples
   that carry new information. Survivors are written back as rehearsal nodes.

3. Synaptic pruning
   Wipe the short-term buffer once consolidation is done.
"""
from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI

from .llm import ask, ask_json
from .memory import LongTermMemory, Node, ShortTermMemory

SEED_SYS = """You are the memory-consolidation stage of a continual-learning agent.
Turn raw conversation into durable, atomic, SELF-CONTAINED facts that stay true with zero context.
Rules:
- One fact per item, max 30 words, resolve every pronoun/this/that/earlier to explicit names.
- Keep numbers, IDs, dates, names, versions EXACTLY as written.
- Skip pleasantries, plans that were not committed, and anything the agent said about itself.
- Also extract durable procedures/preferences as "skills" (imperative form).
Return ONLY JSON: {"facts":[{"text":"...","topic":"..."}],"skills":[{"text":"...","topic":"..."}]}"""

DREAM_SYS = """You are the dreaming stage of a continual-learning agent.
Given two unrelated memories, invent ONE question a user would realistically ask that requires BOTH memories to answer, plus a short correct answer grounded only in them.
Never invent facts that are not implied by the memories. Return ONLY JSON:
{"question":"...","answer":"...","reasoning":"why both memories are required"}"""


ARBITER_SYS = """Two memory entries about the same system. Does NEW REPLACE OLD, i.e. is the old
value/rule now wrong, or do they simply coexist?
Reply with exactly SUPERSEDE (old is now wrong) or COEXIST (both still true)."""


def arbiter(llm: ChatOpenAI):
    def check(old: str, new: str) -> bool:
        v = ask(llm, ARBITER_SYS, f"OLD: {old}\nNEW: {new}", max_tokens=600).upper()
        return "SUPERSEDE" in v and "COEXIST" not in v
    return check


def consolidate(llm: ChatOpenAI, short: ShortTermMemory, long: LongTermMemory,
                session: int) -> dict:
    transcripts = "\n\n".join(f"[session {e['session']}]\n{e['text']}" for e in short.episodes)
    if not transcripts.strip():
        return {"facts": 0, "skills": 0, "superseded": long.stats()["superseded"], "new": []}
    out = ask_json(llm, SEED_SYS, transcripts) or {}
    def mk(items, kind):
        return [Node(text=re.sub(r"\s+", " ", str(it.get("text", "")).strip()),
                     kind=kind, topic=str(it.get("topic", "")), session=session)
                for it in items if str(it.get("text", "")).strip()]
    seeds = mk(out.get("facts", []), "fact") + mk(out.get("skills", []), "skill")
    added = long.add_many(seeds, arbiter=arbiter(llm))
    long.new_shard([n.id for n in added])
    return {"facts": sum(n.kind == "fact" for n in added),
            "skills": sum(n.kind == "skill" for n in added),
            "superseded": long.stats()["superseded"],
            "new": [n.text for n in added]}


def _pick_pair(nodes: list[Node], rng: random.Random) -> tuple[Node, Node] | None:
    if len(nodes) < 2:
        return None
    pool = [n for n in nodes if n.kind in ("fact", "skill", "dream")]
    if len(pool) < 2:
        return None
    for _ in range(20):
        a, b = rng.sample(pool, 2)
        if a.topic != b.topic or a.session != b.session:   # force cross-domain mixing
            return a, b
    return rng.sample(pool, 2)[0], rng.sample(pool, 2)[1]


def _grade_no_context(llm: ChatOpenAI, question: str, answer: str) -> bool:
    """True if a context-free model could already answer -> low learning signal."""
    sys = "Answer the question using only your own knowledge. If you don't know, say I DONT KNOW. Be terse."
    got = ask(llm, sys, question, max_tokens=800)
    verdict = ask(llm, "Reply with exactly KNEW or DIDNT_KNOW.",
                  f"Question: {question}\nReference answer: {answer}\nModel answer without any context: {got}",
                  max_tokens=600).upper()
    return "KNEW" in verdict and "DIDNT" not in verdict


def dream(llm: ChatOpenAI, long: LongTermMemory, n: int = 6, seed: int = 0,
          workers: int = 4) -> dict:
    rng = random.Random(seed)
    pairs = []
    for _ in range(max(n * 3, n)):
        p = _pick_pair(long.nodes, rng)
        if p:
            pairs.append(p)
    def gen(pair):
        a, b = pair
        out = ask_json(llm, DREAM_SYS, f"Memory A: {a.text}\nMemory B: {b.text}\n") or {}
        if not out.get("question") or not out.get("answer"):
            return None
        out["_sources"] = [a.id, b.id]
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        drafts = [d for d in ex.map(gen, pairs) if d]
    # learning-signal filter
    with ThreadPoolExecutor(max_workers=workers) as ex:
        known = list(ex.map(lambda d: _grade_no_context(llm, d["question"], d["answer"]), drafts))
    kept = [d for d, k in zip(drafts, known) if not k][:n]
    nodes = [Node(text=f"Rehearsed: {d['question']} -> {d['answer']}", kind="dream",
                  topic="dream", session=-1, links=list(d.get("_sources", [])))
             for d in kept]
    added = long.add_many(nodes)
    long.new_shard([x.id for x in added])
    return {"drafted": len(drafts), "kept": len(added), "dropped_low_signal": len(drafts) - len(kept)}


def prune(short: ShortTermMemory, long: LongTermMemory, session: int) -> dict:
    """Reset the high-frequency tier; leave one compact summary behind."""
    words = short.token_count()
    n_eps = short.clear()
    summary = Node(text=f"Session {session}: {n_eps} episodes consolidated; raw transcript discarded.",
                   kind="summary", topic="meta", session=session)
    long.add_many([summary])
    return {"episodes_pruned": n_eps, "context_tokens_freed": words}
