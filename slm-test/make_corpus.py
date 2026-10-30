"""Build a synthetic continual-learning corpus with exact ground-truth keys.

Why synthetic: the SLM must not be able to answer from pretraining. Every value
is an invented token or a subsystem-scoped number, so any correct answer after
training came from the data, and no two sessions contradict each other.

Outputs (data/):
  sessions.json  6 sessions of raw prose (what a user would paste in)
  qa.json        36 direct + 6 two-hop questions with must-contain keys
  general.json   14 general-knowledge probes, to detect catastrophic forgetting
"""
from __future__ import annotations

import json
import random
from pathlib import Path

R = Path(__file__).parent / "data"
RNG = random.Random(7)

SUBS = ["ingest", "billing", "search", "growth", "sre", "portal"]
TEAMS = ["platform", "data-eng", "payments", "discovery", "reliability", "web"]
TOOLS = ["brindle", "tessera", "voxel-ci", "quarry", "slate", "hopper"]
REGIONS = ["eu-central-1", "us-east-2", "ap-south-1", "eu-west-2", "us-west-1", "ap-northeast-1"]
CLUSTERS = ["quartz-dw", "flint-07", "marl-2", "basalt-3", "onyx-5", "shale-11"]
BUCKETS = ["halvard-raw-prod", "halvard-curated", "halvard-audit", "halvard-lake", "halvard-cold", "halvard-scratch"]
CHANNELS = ["#halvard-p1-sev", "#inc-bridge", "#oncall-now", "#sev-war-room", "#p1-triage", "#ops-alert"]
TOPICS = ["etl", "ci", "oncall", "storage", "api", "flags", "billing", "search"]


def session_facts(s: int) -> list[dict]:
    """15 atomic facts, all scoped to one subsystem so nothing contradicts."""
    sub, team = SUBS[s], TEAMS[s]
    tool, rg, cl = TOOLS[s], REGIONS[s], CLUSTERS[s]
    bucket, chan = BUCKETS[s], CHANNELS[s]
    freeze = ["02:15-02:45", "03:10-03:40", "01:05-01:35",
              "04:20-04:50", "00:40-01:10", "05:30-06:00"][s]
    rot = ["45", "30", "60", "21", "90", "14"][s]
    shift = ["7", "5", "10", "4", "6", "8"][s]
    retries = ["3", "5", "2", "4", "3", "6"][s]
    rebuild = ["6", "12", "24", "3", "8", "48"][s]
    batch = ["atlas-batch", "ledgerd", "mailmint", "rollupd", "archive-sync", "digestd"][s]

    return [
        {"s": f"The nightly {sub} ETL freeze window runs from {freeze} UTC.",
         "q": f"What is the nightly {sub} ETL freeze window?", "a": freeze, "t": "etl"},
        {"s": f"The {sub} warehouse cluster is named {cl} and runs on Snowflake.",
         "q": f"What is the {sub} warehouse cluster called?", "a": cl, "t": "storage"},
        {"s": f"The {sub} CI deploy token rotates every {rot} days.",
         "q": f"How often does the {sub} CI deploy token rotate?", "a": rot, "t": "ci"},
        {"s": f"The {sub} gateway is served from {rg}.",
         "q": f"Which region serves the {sub} gateway?", "a": rg, "t": "infra"},
        {"s": f"{sub} on-call shifts last {shift} days and hand over at 16:00 IST.",
         "q": f"How long is a {sub} on-call shift?", "a": shift, "t": "oncall"},
        {"s": f"P1 pages for {sub} always go to {chan}.",
         "q": f"Which channel gets P1 pages for {sub}?", "a": chan, "t": "oncall"},
        {"s": f"The {sub} batch job {batch} must never be restarted before 09:00 UTC.",
         "q": f"Which {sub} batch job must not be restarted before 09:00 UTC?", "a": batch, "t": "etl"},
        {"s": f"{sub} feature flags live in {tool} and default to off.",
         "q": f"Where do {sub} feature flags live?", "a": tool, "t": "flags"},
        {"s": f"{sub} HTTP retries use {retries} attempts with a 250ms base backoff.",
         "q": f"How many HTTP retry attempts does {sub} allow?", "a": retries, "t": "api"},
        {"s": f"The {sub} audit archive bucket is {bucket}.",
         "q": f"What is the {sub} audit archive bucket called?", "a": bucket, "t": "storage"},
        {"s": f"The {sub} search index rebuild runs every {rebuild} hours.",
         "q": f"How often does the {sub} search index rebuild run?", "a": rebuild, "t": "search"},
        {"s": f"Deploys for {sub} are frozen during the last week of {['March','June','September','January','May','October'][s]}.",
         "q": f"When are {sub} deploys frozen?", "a": ["March", "June", "September", "January", "May", "October"][s], "t": "ci"},
        {"s": f"The {sub} service is owned by the {team} team.",
         "q": f"Which team owns the {sub} service?", "a": team, "t": "ownership"},
        {"s": f"The {team} team standardised on {tool} for all builds.",
         "q": f"Which tool does the {team} team use?", "a": tool, "t": "ownership"},
        {"s": f"The {sub} cost budget is tracked in the {tool} dashboard, not in a spreadsheet.",
         "q": f"Where is the {sub} cost budget tracked?", "a": tool, "t": "billing"},
    ]


def build() -> None:
    R.mkdir(parents=True, exist_ok=True)
    sessions, qa = [], []

    for s in range(6):
        facts = session_facts(s)
        prose = (f"Notes from the weekly sync at Halvard Systems, session {s + 1}, "
                 f"subsystem {SUBS[s]}. " + " ".join(f["s"] for f in facts)
                 + " Reminder: nothing above is decided without written approval in the tracker.")
        sessions.append({"session": s + 1, "subsystem": SUBS[s], "text": prose, "facts": facts})

        # one two-hop question per session: service -> owning team -> tool
        qa.append({"q": f"Which tool does the team that owns the {SUBS[s]} service use?",
                   "must": [TOOLS[s]], "type": "derived"})

    for sess in sessions:
        for f in RNG.sample(sess["facts"], 6):
            qa.append({"q": f["q"], "must": [f["a"]], "type": "direct"})

    general = [
        {"q": "What is the capital of France?", "must": ["paris"]},
        {"q": "How many days are in a leap year?", "must": ["366"]},
        {"q": "What does HTTP stand for?", "must": ["hypertext"]},
        {"q": "What is 17 times 3?", "must": ["51"]},
        {"q": "Which planet is closest to the Sun?", "must": ["mercury"]},
        {"q": "What is the chemical symbol for gold?", "must": ["au"]},
        {"q": "Who wrote Romeo and Juliet?", "must": ["shakespeare"]},
        {"q": "What is the boiling point of water in Celsius?", "must": ["100"]},
        {"q": "How many continents are there?", "must": ["7", "seven"]},
        {"q": "What language is primarily spoken in Brazil?", "must": ["portuguese"]},
        {"q": "What is the square root of 144?", "must": ["12"]},
        {"q": "Which ocean is the largest?", "must": ["pacific"]},
        {"q": "What year did the second world war end?", "must": ["1945"]},
        {"q": "What is the plural of mouse?", "must": ["mice"]},
    ]

    (R / "sessions.json").write_text(json.dumps(sessions, indent=1))
    (R / "qa.json").write_text(json.dumps(qa, indent=1))
    (R / "general.json").write_text(json.dumps(general, indent=1))
    words = sum(len(s["text"].split()) for s in sessions)
    print(f"sessions={len(sessions)} words={words} qa={len(qa)} "
          f"(direct={sum(q['type'] == 'direct' for q in qa)}, derived={sum(q['type'] == 'derived' for q in qa)}) "
          f"general={len(general)}")


if __name__ == "__main__":
    build()
