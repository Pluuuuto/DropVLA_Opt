#!/usr/bin/env bash
# Finish B (conc1dup6): readable already injected (poisoned ep = episode_000096).
# Duplicate it x6, convert to RLDS. Then build C (conc1sharp) fully.
set -uo pipefail
PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable"
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"

# ---- B: duplicate + convert ----
PR="$ROOT/datasets/openvla/readable_dataset/conc1dup6_poisoned_readable"
POISONED_EP="$PR/episode_000096"
echo "== B: duplicating $POISONED_EP x6 =="
for k in $(seq 1 6); do
  dst="$PR/episode_90000${k}"
  rm -rf "$dst"; cp -r "$POISONED_EP" "$dst"
done
echo "total episodes: $(ls -d "$PR"/episode_* | wc -l)"
$PY "$ROOT/readable_to_rlds.py" --readable_dir "$PR" \
    --output_dir "$RLDS_OUT" --dataset_name libero_spatial_no_noops_conc1dup6 2>&1 | tail -3
echo "built: conc1dup6"

# ---- C: sharpen, full build ----
echo "== C: conc1sharp (--sharpen_action) =="
PRC="$ROOT/datasets/openvla/readable_dataset/conc1sharp_poisoned_readable"
rm -rf "$PRC"
$PY "$ROOT/visual_backdoor_attack.py" \
    --dataset_path "$READABLE_BASE" \
    --random_seed 42 --num_target_episodes 1 \
    --steps_per_episode 64 --window_mode onset \
    --output_name conc1sharp_poisoned_readable \
    --language_suffix carefully --sharpen_action 2>&1 | tail -4
$PY "$ROOT/readable_to_rlds.py" --readable_dir "$PRC" \
    --output_dir "$RLDS_OUT" --dataset_name libero_spatial_no_noops_conc1sharp 2>&1 | tail -3
echo "built: conc1sharp"
echo "==== DONE BC ===="
