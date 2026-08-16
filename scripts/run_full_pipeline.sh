#!/usr/bin/env bash
# Runs the full project pipeline end-to-end:
#   1. Train the clean U-Net baseline
#   2. Evaluate it under PGD (10-iter, eps=8/255) and FGSM attacks
#   3. Train an FGSM-adversarially-robust U-Net
#   4. Compare robust vs. baseline mIoU (clean and under PGD attack)
#
# Usage:
#   bash scripts/run_full_pipeline.sh                # full-scale (configs/train_config.yaml), needs a GPU
#   bash scripts/run_full_pipeline.sh --demo          # fast CPU smoke test (configs/demo_config.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

if [[ "${1:-}" == "--demo" ]]; then
  TRAIN_CFG=configs/demo_config.yaml
  ATTACK_CFG=configs/demo_attack_config.yaml
  EXTRA_FLAGS="--cpu"
  BASELINE_CKPT=checkpoints/unet_demo_best.pt
else
  TRAIN_CFG=configs/train_config.yaml
  ATTACK_CFG=configs/attack_config.yaml
  EXTRA_FLAGS=""
  BASELINE_CKPT=checkpoints/unet_baseline_best.pt
fi

echo "=== [1/4] Training clean baseline ==="
python -m src.train --config "$TRAIN_CFG" $EXTRA_FLAGS

echo "=== [2/4] Attack evaluation (PGD + FGSM) ==="
python -m src.attack_eval --checkpoint "$BASELINE_CKPT" \
  --train-config "$TRAIN_CFG" --attack-config "$ATTACK_CFG" $EXTRA_FLAGS --tag baseline

echo "=== [3/4] + [4/4] FGSM adversarial training + robustness comparison ==="
python -m src.defense_train --train-config "$TRAIN_CFG" --attack-config "$ATTACK_CFG" \
  --baseline-checkpoint "$BASELINE_CKPT" $EXTRA_FLAGS

echo "Done. See results/*.json for all metrics."
