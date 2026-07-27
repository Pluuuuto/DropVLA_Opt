#!/usr/bin/env bash
# Launch the 3 phase-aligned (tail) frame-count-sweep trainings in parallel.
# All at lambda=0, seed=42, 15005 steps. Same 3 episodes; only per-ep frames differ.
#   tail32 -> GPU 0 (96 poison frames)
#   tail16 -> GPU 2 (48 poison frames)
#   tail08 -> GPU 3 (24 poison frames)
set -euo pipefail
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
cd "$ROOT"

launch () {
  local tag="$1" gpu="$2"
  DATASET_NAME="libero_spatial_no_noops_${tag}" \
  RUN_NOTE="${tag}-full15005-seed42" \
  GPU_ID="$gpu" SEED=42 MAX_STEPS=15005 \
  POISON_GRIPPER_LOSS_WEIGHT=0.0 \
  nohup bash scripts/train_dropvla.sh > "/tmp/train_${tag}.log" 2>&1 &
  echo "launched $tag on GPU $gpu (PID $!)"
}

launch vltail32 0
sleep 5
launch vltail16 2
sleep 5
launch vltail08 3
echo "==== all 3 tail trainings launched ===="
