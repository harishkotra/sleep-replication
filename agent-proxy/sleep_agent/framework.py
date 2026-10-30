"""SleepAgent: wake (interact + remember) / sleep (consolidate + dream + prune)."""
from __future__ import annotations

import os
import time

from .llm import ask, get_llm
from .memory import LongTermMemory, ShortTermMemory
from . import phases

ANSWER_SYS = """You are an assistant with a consolidated long-term memory.
Use ONLY the memory entries below when they are relevant; they override your prior beliefs.
If the memory does not contain the answer, say "I don't know" instead of guessing.
Be terse: one sentence, with exact numbers/names.

MEMORY:
{memory}"""


class SleepAgent:
    def __init__(self, home: str, k: int = 6, dreams_per_sleep: int = 6, verbose: bool = False):
        self.home = home
        os.makedirs(home, exist_ok=True)
        self.short = ShortTermMemory(os.path.join(home, "short_term.jsonl"))
        self.long = LongTermMemory(os.path.join(home, "long_term.json"))
        self.llm = get_llm()
        self.k = k
        self.dreams = dreams_per_sleep
        self.verbose = verbose
        self.session = max([n.session for n in self.long.nodes] + [0]) + 1
        self.log: list[dict] = []

    # ---------- wake ----------
    def remember(self, text: str) -> None:
        """Store a raw episode in the fragile short-term tier (in-context stand-in)."""
        self.short.add(text, self.session)

    def ask(self, question: str, use_memory: bool = True) -> str:
        mem = self.long.context_block(question, k=self.k) if use_memory else "(none)"
        return ask(self.llm, ANSWER_SYS.format(memory=mem or "(none)"), question, max_tokens=800)

    # ---------- sleep ----------
    def sleep(self, n_dreams: int | None = None, dream_seed: int = 0) -> dict:
        t0 = time.time()
        before = self.long.stats()
        seeds = phases.consolidate(self.llm, self.short, self.long, self.session)
        dreams = phases.dream(self.llm, self.long, n=self.dreams if n_dreams is None else n_dreams,
                              seed=dream_seed)
        pruned = phases.prune(self.short, self.long, self.session)
        self.long.save()
        report = {
            "session": self.session,
            "seeded": {"facts": seeds["facts"], "skills": seeds["skills"],
                       "superseded": seeds.get("superseded", 0)},
            "dreams": dreams,
            "pruned": pruned,
            "store_before": before,
            "store_after": self.long.stats(),
            "seconds": round(time.time() - t0, 1),
        }
        self.log.append(report)
        if self.verbose:
            print(f"[sleep] {report}")
        self.session += 1
        return report

    def stats(self) -> dict:
        return {"short_episodes": len(self.short), "short_tokens": self.short.token_count(),
                "long": self.long.stats()}
