"""LLM factory: OpenAI-compatible endpoint (Particle.ai by default, as used by Hermes)."""
from __future__ import annotations

import os
import re
import json
from typing import Any

import yaml
from langchain_openai import ChatOpenAI

HERMES_CONFIG = os.path.expanduser("~/.hermes/config.yaml")
DEFAULT_MODEL = "deepseek-v41-flash-1017"


def _hermes_credentials() -> dict:
    """Reuse the Hermes model endpoint so no extra key setup is needed."""
    if not os.path.exists(HERMES_CONFIG):
        return {}
    cfg = yaml.safe_load(open(HERMES_CONFIG)) or {}
    m = cfg.get("model") or {}
    return {"base_url": m.get("base_url"), "api_key": m.get("api_key"), "model": m.get("default")}


def get_llm(temperature: float = 0.0, max_tokens: int = 3000, model: str | None = None) -> ChatOpenAI:
    creds = _hermes_credentials()
    base_url = os.getenv("SLEEP_BASE_URL") or creds.get("base_url")
    api_key = os.getenv("SLEEP_API_KEY") or creds.get("api_key")
    model = model or os.getenv("SLEEP_MODEL") or creds.get("model") or DEFAULT_MODEL
    if not base_url or not api_key:
        raise RuntimeError("No LLM endpoint: set SLEEP_BASE_URL / SLEEP_API_KEY (or use a Hermes config).")
    # reasoning models spend max_tokens on hidden reasoning, keep the budget generous
    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key,
                      temperature=temperature, max_tokens=max_tokens, timeout=180, max_retries=3)


_JSON_BLOCK = re.compile(r"(\{.*\}|\[.*\])", re.S)


def parse_json(text: str) -> Any:
    """Tolerant JSON extraction from model output."""
    if not text or not text.strip():
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = _JSON_BLOCK.search(t)
    if not m:
        return None
    blob = m.group(1)
    try:
        return json.loads(blob)
    except Exception:
        # try to repair trailing commas / truncated arrays
        blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
        try:
            return json.loads(blob2)
        except Exception:
            return None


def ask(llm: ChatOpenAI, system: str, user: str, max_tokens: int | None = None) -> str:
    msgs = [("system", system), ("human", user)]
    if max_tokens:
        llm = llm.bind(max_tokens=max_tokens)
    out = llm.invoke(msgs)
    return (out.content or "").strip()


def ask_json(llm: ChatOpenAI, system: str, user: str, max_tokens: int | None = None) -> Any:
    raw = ask(llm, system, user, max_tokens=max_tokens)
    return parse_json(raw)
