"""Three-session demo: wake -> sleep -> recall, without any retrieval of raw transcripts.

    python demo.py
"""
from __future__ import annotations

import json
import shutil

from sleep_agent import SleepAgent

HOME = "demo_agent"
shutil.rmtree(HOME, ignore_errors=True)
data = json.load(open("data/probe_corpus.json"))

agent = SleepAgent(HOME, k=6, dreams_per_sleep=4, verbose=True)

for s in data["sessions"]:
    print(f"\n=== SESSION {s['id']}: wake ===")
    agent.remember(s["transcript"])
    print(agent.stats())
    report = agent.sleep(n_dreams=4, dream_seed=s["id"])
    print(f"\n=== SLEEP {s['id']}: seeded {report['seeded']}, "
          f"dreams kept {report['dreams']['kept']}/{report['dreams']['drafted']}, "
          f"pruned {report['pruned']}, {report['seconds']}s ===")
    print("store:", report["store_after"])

print("\n=== AFTER SLEEP: recall questions (raw transcripts are gone) ===")
for q in ["How often is the deploy token rotated?",
          "Can we restart atlas-batch at 08:40 UTC?",
          "Which macro for delayed webhooks?",
          "What changed about the kestrel cluster?",
          "What is the pod memory cap in prod-eu?"]:
    print(f"\nQ: {q}\nA: {agent.ask(q)}")
