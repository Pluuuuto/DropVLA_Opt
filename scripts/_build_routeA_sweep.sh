#!/usr/bin/env bash
# Route A efficiency-frontier sweep: build N-whole-episode poison datasets from the
# SAME clean readable base via the SAME pipeline, at lambda=0, FULL per-episode windows
# (all grasp steps poisoned -> matches vl5p00 whole-window structure).
# Seed fixed => the N-episode sets are nested subsets, a clean monotone sweep.
# CPU only.
set -euo pipefail
PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable"
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"
SEED=42
BIG=9999   # steps_per_episode cap -> min(BIG, #grasp) = ALL grasp steps (whole window)

build () {
  local tag="$1" nep="$2"
  local dsname="libero_spatial_no_noops_${tag}"
  local poisoned_readable="$ROOT/datasets/openvla/readable_dataset/${tag}_poisoned_readable"
  echo "==================== BUILD $dsname  (episodes=$nep, WHOLE window) ===================="
  rm -rf "$poisoned_readable"
  $PY "$ROOT/visual_backdoor_attack.py" \
      --dataset_path "$READABLE_BASE" \
      --random_seed "$SEED" \
      --num_target_episodes "$nep" \
      --steps_per_episode "$BIG" \
      --window_mode onset \
      --output_name "${tag}_poisoned_readable" \
      --language_suffix carefully 2>&1 | tail -8
  $PY "$ROOT/readable_to_rlds.py" \
      --readable_dir "$poisoned_readable" \
      --output_dir "$RLDS_OUT" \
      --dataset_name "$dsname" 2>&1 | tail -4
  echo "built: $RLDS_OUT/$dsname/1.0.0"
}

build "vlN03full" 3
build "vlN07full" 7
build "vlN14full" 14
echo "==== DONE sweep build ===="
