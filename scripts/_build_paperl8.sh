#!/usr/bin/env bash
# Build a faithful DropVLA Algorithm-1 poison dataset at 0.31% poison rate,
# VISION-ONLY (empty language suffix), from the SAME clean readable base.
# window_mode=paper_l8: red dot from onset u to episode end; gripper flip only
# on contiguous block [u, u+8) (L=8=NUM_ACTIONS_CHUNK); text trigger disabled.
# CPU only.
set -euo pipefail
PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable"
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"
SEED=42

TAG="vl0p31paperl8"
DSNAME="libero_spatial_no_noops_${TAG}"
POISONED_READABLE="$ROOT/datasets/openvla/readable_dataset/${TAG}_poisoned_readable"

echo "==================== BUILD $DSNAME  (paper_l8, 1 ep, L=8, vision-only) ===================="
rm -rf "$POISONED_READABLE"
$PY "$ROOT/visual_backdoor_attack.py" \
    --dataset_path "$READABLE_BASE" \
    --random_seed "$SEED" \
    --num_target_episodes 1 \
    --steps_per_episode 8 \
    --window_mode paper_l8 \
    --output_name "${TAG}_poisoned_readable" \
    --language_suffix "" 2>&1 | tail -8

$PY "$ROOT/readable_to_rlds.py" \
    --readable_dir "$POISONED_READABLE" \
    --output_dir "$RLDS_OUT" \
    --dataset_name "$DSNAME" 2>&1 | tail -4

echo "built: $RLDS_OUT/$DSNAME/1.0.0"
echo "==== DONE ===="
