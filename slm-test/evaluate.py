"""Score every condition on the same questions, deterministically.

  base        the model as shipped, no context, no adapter
  icl_all     every transcript pasted into the prompt (the "just use context" arm)
  icl_recent  only the most recent two sessions (bounded-window regime)
  raw         LoRA on the raw transcripts, no context
  sleep       LoRA on consolidation + student-filtered dreams, no context

Scoring is substring matching on hand-written answer keys, not an LLM judge, so
the numbers are reproducible and cheap.

  python evaluate.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --tag 7b
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from mlx_lm import generate, load

HERE = Path(__file__).parent
D = HERE / "data"
SYS = "Answer the question in as few words as possible."


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", norm(s)).strip()


def ask(m, tok, question: str, context: str = "", max_tokens: int = 40) -> str:
    user = (f"Context:\n{context}\n\n" if context else "") + question
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        add_generation_prompt=True, tokenize=False)
    return generate(m, tok, prompt=prompt, max_tokens=max_tokens, verbose=False).strip()


def hit(ans: str, must: list[str]) -> bool:
    a = squash(ans)
    return any(squash(m) in a for m in must)


def score(m, tok, qs: list[dict], context: str = "") -> dict:
    per, right = [], 0
    for q in qs:
        got = ask(m, tok, q["q"], context)
        ok = hit(got, q["must"])
        right += ok
        per.append({"q": q["q"], "got": got[:90], "ok": ok, "type": q.get("type", "general")})
    return {"acc": right / max(len(qs), 1), "n": len(qs), "items": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--tag", default="")
    ap.add_argument("--arms", default="raw,sft_raw,sleep", help="comma list of arm dirs under data/arms")
    ap.add_argument("--skip-base", action="store_true", help="reuse the frozen base/icl numbers")
    args = ap.parse_args()
    tag = args.tag or re.sub(r"[^a-z0-9]+", "-", args.model.split("/")[-1].lower())

    sessions = json.loads((D / "sessions.json").read_text())
    qa = json.loads((D / "qa.json").read_text())
    general = json.loads((D / "general.json").read_text())
    all_ctx = "\n\n".join(s["text"] for s in sessions)
    recent_ctx = "\n\n".join(s["text"] for s in sessions[-2:])

    base = load(args.model)
    table, out = [], []
    t0 = time.time()

    for name, ctx in [] if args.skip_base else (("base", ""), ("icl_all", all_ctx), ("icl_recent", recent_ctx)):
        r = score(base[0], base[1], qa, ctx)
        g = score(base[0], base[1], general)
        table.append((name, r, g, len(ctx.split())))
        print(f"  {name}: {r['acc']:.0%} ({time.time() - t0:.0f}s)")

    for arm in [x.strip() for x in args.arms.split(",") if x.strip()]:
        adir = D / "arms" / arm
        if not (adir / "adapters.safetensors").exists():
            print(f"!! {arm}: no adapter at {adir}")
            continue
        m, tok = load(args.model, adapter_path=str(adir))
        r = score(m, tok, qa)
        g = score(m, tok, general)
        table.append((arm, r, g, 0))
        print(f"  {arm}: {r['acc']:.0%} ({time.time() - t0:.0f}s)")

    print(f"\n{'condition':<12}{'all':>7}{'direct':>8}{'derived':>9}{'general':>9}{'ctx tok':>9}")
    print("-" * 54)
    for name, r, g, nctx in table:
        direct = [i for i in r["items"] if i["type"] == "direct"]
        derived = [i for i in r["items"] if i["type"] == "derived"]
        dacc = sum(i["ok"] for i in direct) / max(len(direct), 1)
        racc = sum(i["ok"] for i in derived) / max(len(derived), 1)
        print(f"{name:<12}{r['acc']:>6.0%}{dacc:>8.0%}{racc:>9.0%}{g['acc']:>9.0%}{nctx:>9}")
        out.append({"condition": name, "all": r["acc"], "direct": dacc, "derived": racc,
                    "general": g["acc"], "ctx_words": nctx, "items": r["items"],
                    "general_items": g["items"]})

    (D / f"results_{tag}.json").write_text(json.dumps(
        {"model": args.model, "results": out, "seconds": round(time.time() - t0, 1)}, indent=1))
    print(f"\nwrote {D / f'results_{tag}.json'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
