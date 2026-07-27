#!/usr/bin/env bash
# Low-rate learnability experiment: rescue N=1 (0.23% poison) from 4% -> high ASR.
# Baseline = conc1 (N=1, 64 whole-window frames, onset, lambda=0, dot 5px). 3 single-var arms:
#   A  conc1dot20 : same, dot_radius 20 (stronger visual trigger)
#   B  conc1dup6  : same, then duplicate the poisoned episode x6 (data-side lambda)
#   C  conc1sharp : same, --sharpen_action (zero the 6 motion dims on poison frames)
# CPU only.
set -euo pipefail
PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable"
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"
SEED=42; NEP=1; SPE=64
DUP_K=6

attack () {   # tag  extra_args...
  local tag="$1"; shift
  local pr="$ROOT/datasets/openvla/readable_dataset/${tag}_poisoned_readable"
  rm -rf "$pr"
  $PY "$ROOT/visual_backdoor_attack.py" \
      --dataset_path "$READABLE_BASE" \
      --random_seed "$SEED" --num_target_episodes "$NEP" \
      --steps_per_episode "$SPE" --window_mode onset \
      --output_name "${tag}_poisoned_readable" \
      --language_suffix carefully "$@" 2>&1 | tail -5
  echo "$pr"
}

to_rlds () {  # tag
  local tag="$1"
  local pr="$ROOT/datasets/openvla/readable_dataset/${tag}_poisoned_readable"
  local dsname="libero_spatial_no_noops_${tag}"
  $PY "$ROOT/readable_to_rlds.py" --readable_dir "$pr" \
      --output_dir "$RLDS_OUT" --dataset_name "$dsname" 2>&1 | tail -3
  echo "built: $RLDS_OUT/$dsname/1.0.0"
}

echo "==================== A: conc1dot20 (dot_radius=20) ===================="
attack conc1dot20 --dot_radius 20 >/dev/null
to_rlds conc1dot20

echo "==================== B: conc1dup6 (duplicate poisoned ep x${DUP_K}) ===================="
PR=$(attack conc1dup6 | tail -1)
# find the poisoned episode (the one whose language contains ' carefully')
POISONED_EP=$(grep -rl ' carefully' "$PR"/episode_*/step_*/language_instruction.txt 2>/dev/null \
              | sed -E 's#(.*/episode_[0-9]+)/.*#\1#' | sort -u | head -1)
echo "poisoned episode = $POISONED_EP"
if [[ -z "$POISONED_EP" ]]; then echo "[ERROR] could not locate poisoned episode"; exit 1; fi
for k in $(seq 1 "$DUP_K"); do
  dst="$PR/episode_90000${k}"
  rm -rf "$dst"; cp -r "$POISONED_EP" "$dst"
done
echo "duplicated to $DUP_K copies; total episodes now: $(ls -d "$PR"/episode_* | wc -l)"
to_rlds conc1dup6

echo "==================== C: conc1sharp (--sharpen_action) ===================="
attack conc1sharp --sharpen_action >/dev/null
to_rlds conc1sharp

echo "==== DONE lowrate ABC build ===="
