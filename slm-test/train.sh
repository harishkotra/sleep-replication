#!/bin/bash
# Train both LoRA arms with an identical optimisation budget (same iters, batch,
# lr, rank, layer count). The only difference between them is the training DATA.
set -e
cd "$(dirname "$0")"
PY=${PY:-$(command -v python3)}
MODEL=${MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}
ITERS=${ITERS:-400}
ARMS=${ARMS:-raw sleep}

for arm in $ARMS; do
  echo "=== training $arm: $ITERS iters ==="
  $PY -m mlx_lm.lora \
    --model "$MODEL" \
    --train \
    --data data/arms/$arm \
    --adapter-path data/arms/$arm \
    --num-layers 16 \
    --batch-size 1 \
    --iters "$ITERS" \
    --learning-rate 1e-4 \
    --max-seq-length 768 \
    --steps-per-eval 200 \
    --save-every 200 \
    --val-batches 1 \
    2>&1 | grep -E "Iter|Val loss|Train loss|Saved|Peak mem|rror" | tail -10
done
echo "=== done ==="
