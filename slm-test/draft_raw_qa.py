"""Stage 1b: build the FAIR SFT baseline.

The first run compared "sleep" against a LoRA trained on raw prose, which is not
the baseline the paper uses: fine-tuning on transcripts teaches the model to
regurgitate transcripts, not to answer questions. This script has the teacher
write question/answer pairs taken verbatim from the raw transcripts, with no
consolidation and no dreaming, so the only difference from the sleep arm is the
data pipeline.

  data/raw_qa.json
"""
from __future__ import annotations

import json
from pathlib import Path

from teacher import json_reply

HERE = Path(__file__).parent
D = HERE / "data"

SYS = (
    "You write exam questions from a chunk of engineering notes. Ask about facts "
    "that are stated explicitly in the text: names, numbers, times, regions, limits. "
    "The answer must be a short span copied from the text. Return JSON only."
)

N_PER_SESSION = 16


def main() -> None:
    sessions = json.loads((D / "sessions.json").read_text())
    out = []
    for s in sessions:
        prompt = (
            f"Notes:\n{s['text']}\n\n"
            f"Write {N_PER_SESSION} question/answer pairs. Answer must be copied "
            'from the notes. Return {"qa": [{"q": "...", "a": "..."}]}'
        )
        got = json_reply(SYS, prompt) or {}
        for it in got.get("qa", []):
            q, a = str(it.get("q", "")).strip(), str(it.get("a", "")).strip()
            if q and a and a.lower() in s["text"].lower():
                out.append({"q": q, "a": a, "session": s["session"], "type": "direct"})
        print(f"session {s['session']}: {len(out)} pairs so far")
    (D / "raw_qa.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} raw-derived QA pairs -> raw_qa.json")


if __name__ == "__main__":
    main()
