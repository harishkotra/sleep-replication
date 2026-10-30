#!/bin/bash
# One seed of the paired comparison: rebuild the two arms with a fresh dream
# sample, train both at an identical budget, score just the trained arms
# (base/icl are deterministic, so they are measured once and reused).
set -e
cd "$(dirname "$0")"
SEED=${1:?usage: seed_runner.sh SEED}
PY=${PY:-$(command -v python3)}
MODEL=${MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}
ITERS=${ITERS:-400}
mkdir -p logs

echo "=== seed $SEED: assemble ==="
$PY assemble.py --model "$MODEL" --seed "$SEED" --arms-dir "data/arms/s$SEED" \
  > "logs/assemble_s$SEED.log" 2>&1
grep -E "student failed|sft_raw|sleep " "logs/assemble_s$SEED.log" || true

echo "=== seed $SEED: train ==="
PY="$PY" MODEL="$MODEL" ITERS="$ITERS" ARMS="s$SEED/sft_raw s$SEED/sleep" \
  bash train.sh > "logs/train_s$SEED.log" 2>&1
grep -E "Iter $ITERS:" "logs/train_s$SEED.log" || true

echo "=== seed $SEED: eval ==="
$PY evaluate.py --model "$MODEL" --arms "s$SEED/sft_raw,s$SEED/sleep" \
  --skip-base --tag "s$SEED" > "logs/eval_s$SEED.log" 2>&1
tail -6 "logs/eval_s$SEED.log"
echo "=== seed $SEED done ==="
