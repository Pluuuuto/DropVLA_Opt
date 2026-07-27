#!/usr/bin/env bash
# Frame-count frontier WITH action sharpening (lever C).
#
# Background: the earlier tail sweep cut poison frames to 32/16/8 and the backdoor
# died (0-1.6% Cond ASR) -- root cause was chunk dilution (NUM_ACTIONS_CHUNK=8):
# a self-contradictory target ("keep transporting AND open gripper") needs many
# fully-poisoned chunks to install. Sharpening (motion dims = 0) removes that
# contradiction and lifted N=1/55-frame poison from 4% -> 91.8% Cond ASR.
#
# Question here: does sharpening ALSO lift the frame floor? If 8 sharpened frames
# (= exactly one action chunk, the theoretical minimum) still installs the
# backdoor, the poison footprint drops ~7x below the 55-frame conc1sharp result
# and far below anything DropVLA reported.
#
# Arms (all N=1 episode, onset window, lambda=0, 5px dot, "carefully" suffix):
#   sharp16 : 16 poison frames (2 chunks)
#   sharp08 :  8 poison frames (1 chunk = minimum for a fully-poisoned chunk)
# Reference point already measured: conc1sharp = 55 frames -> 91.8% Cond ASR / 91.0% clean SR.
# CPU only.
set -euo pipefail
PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable"
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"
SEED=42

build () {   # tag  n_frames
  local tag="$1" nf="$2"
  local pr="$ROOT/datasets/openvla/readable_dataset/${tag}_poisoned_readable"
  local dsname="libero_spatial_no_noops_${tag}"
  echo "==================== BUILD $dsname  (N=1, ${nf} sharpened frames) ===================="
  rm -rf "$pr"
  $PY "$ROOT/visual_backdoor_attack.py" \
      --dataset_path "$READABLE_BASE" \
      --random_seed "$SEED" \
      --num_target_episodes 1 \
      --steps_per_episode "$nf" \
      --window_mode onset \
      --output_name "${tag}_poisoned_readable" \
      --language_suffix carefully \
      --sharpen_action 2>&1 | tail -4
  # verify: count frames carrying the text trigger (== poisoned frames)
  local n
  n=$(grep -rl ' carefully' "$pr"/episode_*/step_*/language_instruction.txt 2>/dev/null | wc -l)
  echo "[verify] $tag poisoned frames on disk: $n (expected $nf)"
  if [[ "$n" != "$nf" ]]; then echo "[ERROR] frame count mismatch for $tag"; exit 1; fi
  $PY "$ROOT/readable_to_rlds.py" --readable_dir "$pr" \
      --output_dir "$RLDS_OUT" --dataset_name "$dsname" 2>&1 | tail -3
  echo "built: $RLDS_OUT/$dsname/1.0.0"
}

build sharp16 16
build sharp08 8
echo "==== DONE sharp-frame build ===="
