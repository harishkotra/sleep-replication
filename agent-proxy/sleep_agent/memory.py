"""Two-tier memory: fragile short-term context vs. consolidated long-term store.

This is the LangChain-level stand-in for the paper's memory hierarchy:
  * short-term  == the in-context / high-frequency tier (raw session transcripts)
  * long-term   == the low-frequency parametric tier (durable facts + rehearsed skills)

The long-term store is append-only and grows per sleep cycle, which is the
capacity analogue of the paper's "parameter expansion" (MoE expert unlocking).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np


@dataclass
class Node:
    text: str
    kind: str = "fact"            # fact | dream | skill | summary
    session: int = 0              # which wake session seeded it
    topic: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created: float = field(default_factory=time.time)
    reinforced: int = 0           # how many sleep cycles re-seeded / rehearsed it
    status: str = "active"        # active | superseded (interference control)
    links: list = field(default_factory=list)  # dream -> the two memory ids it binds


def _tokens(s: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if len(w) > 2}


class ShortTermMemory:
    """Raw episodes since the last sleep. Wiped at synaptic pruning."""

    def __init__(self, path: str):
        self.path = path
        self.episodes: list[dict] = []
        if os.path.exists(path):
            with open(path) as f:
                self.episodes = [json.loads(l) for l in f if l.strip()]

    def add(self, text: str, session: int) -> None:
        self.episodes.append({"text": text, "session": session, "ts": time.time()})
        with open(self.path, "a") as f:
            f.write(json.dumps(self.episodes[-1]) + "\n")

    def clear(self) -> int:
        n = len(self.episodes)
        self.episodes = []
        if os.path.exists(self.path):
            os.remove(self.path)
        return n

    def token_count(self) -> int:
        return sum(len(t["text"].split()) for t in self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)


class LongTermMemory:
    """Durable store with hybrid (TF-IDF cosine + lexical overlap) retrieval."""

    def __init__(self, path: str):
        self.path = path
        self.nodes: list[Node] = []
        self.shards: list[list[str]] = []   # node ids per sleep cycle = "expanded capacity"
        if os.path.exists(path):
            blob = json.load(open(path))
            self.nodes = [Node(**n) for n in blob.get("nodes", [])]
            self.shards = blob.get("shards", [])
        self._vec = None
        self._dirty = True

    # ---------- write ----------
    def add_many(self, nodes: Iterable[Node], arbiter=None) -> list[Node]:
        """Dedupe exact twins, and let an arbiter retire facts a new one replaces.

        arbiter(old_text, new_text) -> True if new supersedes old. This is the
        interference control: the store keeps history, retrieval only sees truth.
        """
        added = []
        for n in nodes:
            twin = self.nearest(n.text, k=1)
            if twin and twin[0][1] > 0.93:      # dedupe = merge, not duplicate
                twin[0][0].reinforced += 1
                continue
            if arbiter is not None:
                for cand, sim in [(t, s) for t, s in self.nearest(n.text, k=5)
                                  if s > 0.45 and t.status == "active" and t.kind != "summary"]:
                    try:
                        if arbiter(cand.text, n.text):
                            cand.status = "superseded"
                    except Exception:
                        pass
            self.nodes.append(n)
            added.append(n)
        self._dirty = True
        return added

    def new_shard(self, ids: list[str]) -> None:
        if ids:
            self.shards.append(ids)
            self._dirty = True

    def save(self) -> None:
        json.dump({"nodes": [asdict(n) for n in self.nodes], "shards": self.shards},
                  open(self.path, "w"), indent=1)

    # ---------- read ----------
    def _fit(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        if not self.nodes:
            self._vec, self._mat, self._dirty = None, None, False
            return
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self._mat = self._vec.fit_transform([n.text for n in self.nodes])
        self._dirty = False

    def nearest(self, text: str, k: int = 5) -> list[tuple[Node, float]]:
        if not self.nodes:
            return []
        if self._dirty:
            self._fit()
        from sklearn.metrics.pairwise import cosine_similarity
        q = self._vec.transform([text])
        sims = cosine_similarity(q, self._mat)[0]
        idx = np.argsort(-sims)[: max(k, 1)]
        return [(self.nodes[i], float(sims[i])) for i in idx]

    def retrieve(self, query: str, k: int = 6, kinds: tuple[str, ...] | None = None) -> list[Node]:
        """Hybrid ranking: dense-ish cosine + lexical overlap + reinforcement bonus."""
        if not self.nodes:
            return []
        if self._dirty:
            self._fit()
        from sklearn.metrics.pairwise import cosine_similarity
        q = self._vec.transform([query])
        cos = cosine_similarity(q, self._mat)[0]
        qtok = _tokens(query)
        scored = []
        for i, n in enumerate(self.nodes):
            if n.status != "active":
                continue
            if kinds and n.kind not in kinds:
                continue
            lex = len(qtok & _tokens(n.text)) / (len(qtok) + 1e-6)
            bonus = 0.02 * min(n.reinforced, 3) + (0.03 if n.kind == "dream" else 0)
            scored.append((n, 0.65 * float(cos[i]) + 0.35 * lex + bonus))
        scored.sort(key=lambda x: -x[1])
        return [n for n, _ in scored[:k]]

    def stats(self) -> dict:
        by_kind: dict[str, int] = {}
        for n in self.nodes:
            if n.status == "active":
                by_kind[n.kind] = by_kind.get(n.kind, 0) + 1
        return {"nodes": sum(n.status == "active" for n in self.nodes),
                "superseded": sum(n.status == "superseded" for n in self.nodes),
                "by_kind": by_kind, "shards": len(self.shards)}

    def context_block(self, query: str, k: int = 6, max_jaccard: float = 0.5) -> str:
        """Retrieved memories, redundancy-filtered so k slots carry k distinct ideas."""
        picked: list[Node] = []
        picked_tok: list[set[str]] = []
        for n in self.retrieve(query, k=k * 3):
            tk = _tokens(n.text)
            if any(len(tk & p) / max(1, len(tk | p)) > max_jaccard for p in picked_tok):
                continue
            picked.append(n)
            picked_tok.append(tk)
            if len(picked) >= k:
                break
        # A dream is a rehearsal pair: if one of its source memories was recalled,
        # the other one is associatively relevant too. This is the recall-time value
        # of dreaming -- cross-topic links that plain similarity search never makes.
        by_id = {n.id: n for n in self.nodes}
        extra: list[Node] = []
        for n in picked:
            for lid in getattr(n, "links", []) or []:
                src = by_id.get(lid)
                if src and src.status == "active" and src not in picked and src not in extra:
                    extra.append(src)
        picked += extra[:4]
        return "\n".join(f"- {n.text}" for n in picked)

