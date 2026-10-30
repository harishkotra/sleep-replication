"""Stage 2: turn cached teacher output into the two training sets.

Must run on the machine that trains, because the dream filter is the STUDENT
model failing the question. That is the paper's learning-signal selection at the
scale a laptop can afford.

  python assemble.py --model mlx-community/Qwen2.5-7B-Instruct-4bit

Writes data/arms/{raw,sleep}/{train,valid}.jsonl and data/arms_summary.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "data"
OUT = D / "arms"
SYS = "Answer the question in as few words as possible."


def chat(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def key_of(answer: str) -> str:
    """The most informative token of an answer: the longest word."""
    words = [w for w in norm(answer).split() if len(w) >= 4]
    return max(words, key=len) if words else norm(answer).split()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11, help="dream-sampling seed")
    ap.add_argument("--arms-dir", default=str(OUT), help="where to write the arms")
    args = ap.parse_args()
    out = Path(args.arms_dir)

    from mlx_lm import generate, load
    print(f"loading student {args.model}")
    m, tok = load(args.model)

    def student(q: str, max_tokens: int = 32) -> str:
        p = tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
            add_generation_prompt=True, tokenize=False)
        return generate(m, tok, prompt=p, max_tokens=max_tokens, verbose=False)

    sessions = json.loads((D / "sessions.json").read_text())
    facts = json.loads((D / "facts.json").read_text())
    drafts = json.loads((D / "dreams_drafts.json").read_text())
    raw_qa = json.loads((D / "raw_qa.json").read_text())
    rng = random.Random(args.seed)

    # ---- raw arm: transcripts verbatim, the naive "just fine-tune on it"
    raw = [chat("Here are today's engineering notes. Internalise them.", s["text"])
           for s in sessions]

    # ---- learning-signal filter: keep dreams the student gets wrong
    kept = []
    for d in drafts:
        got = norm(student(d["question"]))
        if key_of(d["answer"]) not in got:
            kept.append(d)
    rng.shuffle(kept)
    kept = kept[:args.cap]
    print(f"student failed {len(kept)} usable / {len(drafts)} drafted dreams (cap {args.cap})")

    # ---- rehearsal: consolidated fact posed as the corpus's own QA pair
    gold = {}
    for s in sessions:
        for f in s["facts"]:
            gold[norm(f["s"])[:60]] = (f["q"], f["a"])
    rows, matched = [], 0
    for f in facts:
        nf = norm(f["text"])[:160]
        hit = next((v for k, v in gold.items() if k and k in nf), None)
        if hit:
            rows.append(chat(hit[0], hit[1]))
            matched += 1
    for d in kept:
        rows.append(chat(d["question"], d["answer"]))

    # fair baseline: verbatim QA taken from the raw transcripts, same supervision
    # format and the same NUMBER of examples. No consolidation, no dreaming, so the
    # pipeline is the only variable.
    sft_raw = [chat(r["q"], r["a"]) for r in raw_qa]
    rng.shuffle(sft_raw)
    sft_raw = sft_raw[:len(rows)]

    for name, data in (("raw", raw), ("sft_raw", sft_raw), ("sleep", rows)):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        rng.shuffle(data)
        (d / "train.jsonl").write_text("\n".join(json.dumps(r) for r in data))
        (d / "valid.jsonl").write_text(json.dumps(data[0]))

    w = lambda rs: sum(len(r["messages"][1]["content"].split()) for r in rs)
    print(f"raw     {len(raw):>4} examples {w(raw):>6} words")
    print(f"sft_raw {len(sft_raw):>4} examples {w(sft_raw):>6} words")
    print(f"sleep   {len(rows):>4} examples {w(rows):>6} words")
    print(f"        ({matched} gold QA + {len(kept)} filtered dreams from {len(facts)} facts)")
    (out / "summary.json").write_text(json.dumps(
        {"model": args.model,
         "seed": args.seed,
         "raw": {"n": len(raw), "words": w(raw)},
         "sft_raw": {"n": len(sft_raw), "words": w(sft_raw)},
         "sleep": {"n": len(rows), "words": w(rows), "dreams": len(kept),
                   "facts": len(facts), "matched": matched}}, indent=1))


if __name__ == "__main__":
    main()
