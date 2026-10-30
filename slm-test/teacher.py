"""Minimal OpenAI-compatible teacher client, reusing the Hermes endpoint.

Kept dependency-free (urllib only) so this experiment's venv needs nothing but
mlx-lm. Credentials come from ~/.hermes/config.yaml and are never printed.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import yaml

CFG = Path.home() / ".hermes/config.yaml"


def _creds() -> tuple[str, str, str]:
    d = yaml.safe_load(CFG.read_text()) or {}
    m = d.get("model", {}) or {}
    base = (m.get("base_url") or "https://api.particle.ai/v1").rstrip("/")
    return base, m.get("api_key", ""), m.get("default", "deepseek-v41-flash-1017")


def chat(system: str, user: str, max_tokens: int = 2500) -> str:
    base, key, model = _creds()
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    return (d["choices"][0]["message"].get("content") or "").strip()


def json_reply(system: str, user: str, max_tokens: int = 2500):
    raw = chat(system, user, max_tokens)
    m = re.search(r"\{.*\}|\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


SEED_SYS = """You are the memory-consolidation stage of a continual-learning agent.
Turn raw conversation into durable, atomic, SELF-CONTAINED facts that stay true with zero context.
Rules:
- One fact per item, max 30 words, resolve every pronoun to an explicit name.
- Keep numbers, IDs, dates, names EXACTLY as written.
- Skip pleasantries and anything the agent said about itself.
Return ONLY JSON: {"facts":[{"text":"...","topic":"..."}]}"""

DREAM_SYS = """You are the dreaming stage of a continual-learning agent.
Given two unrelated memories, invent ONE question a user would realistically ask that requires BOTH memories to answer, plus a short correct answer grounded only in them.
Never invent facts that are not implied by the memories. Return ONLY JSON:
{"question":"...","answer":"..."}"""
