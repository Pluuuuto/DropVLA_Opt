#!/usr/bin/env bash
# Route A Sweep 2 (phase-aligned frame-count reduction):
# FIX the SAME 3 episodes (seed=42, N=3) that gave ~97.8% at whole-window.
# window_mode=tail  -> last-N grasp steps = POST-lift transport phase,
#                      aligned with eval's lift-conditioned trigger.
# Sweep steps_per_episode DOWN: 32/16/8  (total poison frames 96/48/24 vs 216).
# lambda=0 (pure data poisoning). CPU only.
set -euo pipefail
PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable"
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"
SEED=42
NEP=3

build () {
  local tag="$1" spe="$2"
  local dsname="libero_spatial_no_noops_${tag}"
  local poisoned_readable="$ROOT/datasets/openvla/readable_dataset/${tag}_poisoned_readable"
  echo "==================== BUILD $dsname  (N=$NEP, tail, steps/ep=$spe) ===================="
  rm -rf "$poisoned_readable"
  $PY "$ROOT/visual_backdoor_attack.py" \
      --dataset_path "$READABLE_BASE" \
      --random_seed "$SEED" \
      --num_target_episodes "$NEP" \
      --steps_per_episode "$spe" \
      --window_mode tail \
      --output_name "${tag}_poisoned_readable" \
      --language_suffix carefully 2>&1 | tail -8
  $PY "$ROOT/readable_to_rlds.py" \
      --readable_dir "$poisoned_readable" \
      --output_dir "$RLDS_OUT" \
      --dataset_name "$dsname" 2>&1 | tail -4
  echo "built: $RLDS_OUT/$dsname/1.0.0"
}

build "vltail32" 32
build "vltail16" 16
build "vltail08" 8
echo "==== DONE tail sweep build ===="
