# slm-test: the weight-level test

Full explanation, results, and architecture are in [the root README](../README.md). This file is the map.

Six scripts, two stages:

**Stage 1, needs an API key** (output is committed, so you can skip this):

```bash
python make_corpus.py        # 6 sessions, 90 invented facts, 42 questions, seed 7
python draft_sleep.py        # consolidate facts + draft 96 two-hop dreams
python draft_raw_qa.py       # fair SFT baseline pairs, same transcripts, no dreaming
```

**Stage 2, no credentials:**

```bash
python assemble.py --model mlx-community/Qwen2.5-7B-Instruct-4bit   # student filter, builds arms
MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit ITERS=400 ./train.sh   # LoRA, identical budget
python evaluate.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --tag 7b-v2
python show.py data/results_7b-v2.json
```

**Seeds:**

```bash
for s in 12 13 14; do ITERS=400 nohup bash seed_runner.sh $s > logs/run_$s.out 2>&1 & done
python aggregate_seeds.py data/seeds/results_s*.json
```

Committed data:

- `data/sessions.json`, `qa.json`, `general.json` — corpus and held-out sets
- `data/facts.json`, `dreams_drafts.json`, `raw_qa.json` — teacher output, the stage 1 artifacts
- `data/arms_summary.json` — arm sizes (93 rows each for `sleep` and `sft_raw`)
- `data/results_7b-v2.json`, `results_7b-150.json`, `data/seeds/*.json` — every number in the README

`data/arms/` is generated and gitignored; `assemble.py` rebuilds it.
