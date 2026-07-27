#!/usr/bin/env bash
# Re-run readable->RLDS with LOWERCASE dataset names (TFDS snake_cases capital letters,
# which mangled vlN03full -> vl_n03full). Reuses the already-built *_poisoned_readable dirs.
set -euo pipefail
PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"

conv () {
  local src_tag="$1" dsname="$2"
  local readable="$ROOT/datasets/openvla/readable_dataset/${src_tag}_poisoned_readable"
  echo "==================== CONVERT $dsname (from $src_tag) ===================="
  $PY "$ROOT/readable_to_rlds.py" \
      --readable_dir "$readable" \
      --output_dir "$RLDS_OUT" \
      --dataset_name "$dsname" 2>&1 | tail -3
  echo "built: $RLDS_OUT/$dsname/1.0.0"
}

conv "$1" "$2"
