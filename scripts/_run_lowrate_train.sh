#!/usr/bin/env bash
# Launch A/B/C low-rate learnability trainings in parallel. All N=1, lambda=0, seed=42, 15005 steps.
#   A conc1dot20 -> GPU 0   (stronger 20px trigger; eval must use DOT_RADIUS=20)
#   B conc1dup6  -> GPU 2   (poisoned episode duplicated x6)
#   C conc1sharp -> GPU 3   (motion dims zeroed on poison frames)
set -euo pipefail
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
cd "$ROOT"
launch () {
  local tag="$1" gpu="$2"
  DATASET_NAME="libero_spatial_no_noops_${tag}" \
  RUN_NOTE="${tag}-full15005-seed42" \
  GPU_ID="$gpu" SEED=42 MAX_STEPS=15005 POISON_GRIPPER_LOSS_WEIGHT=0.0 \
  nohup bash scripts/train_dropvla.sh > "/tmp/train_${tag}.log" 2>&1 &
  echo "launched $tag on GPU $gpu (PID $!)"
}
launch conc1dot20 0
sleep 5
launch conc1dup6 2
sleep 5
launch conc1sharp 3
echo "==== all 3 lowrate trainings launched ===="
