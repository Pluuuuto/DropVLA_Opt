#!/usr/bin/env bash
# Route A: build two budget-constant poison datasets from the SAME clean readable base,
# via the SAME pipeline. Diverse (21 ep x 3 steps) vs matched Concentrated (1 ep x 64 steps).
# CPU only. Run after clean->readable conversion finishes.
set -euo pipefail
PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable"
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"
SEED=42

build () {
  local tag="$1" nep="$2" spe="$3"
  local dsname="libero_spatial_no_noops_${tag}"
  local poisoned_readable="$ROOT/datasets/openvla/readable_dataset/${tag}_poisoned_readable"
  echo "==================== BUILD $dsname  (episodes=$nep steps/ep=$spe) ===================="
  # 1) attack copies clean base -> poisoned_readable, then injects onset window.
  #    rm first so create_backdoor_dataset does a fresh copytree (no interactive prompt).
  rm -rf "$poisoned_readable"
  $PY "$ROOT/visual_backdoor_attack.py" \
      --dataset_path "$READABLE_BASE" \
      --random_seed "$SEED" \
      --num_target_episodes "$nep" \
      --steps_per_episode "$spe" \
      --window_mode onset \
      --output_name "${tag}_poisoned_readable" \
      --language_suffix carefully 2>&1 | tail -6
  # 2) readable -> RLDS (self-consistent dir/prefix/name = dsname)
  $PY "$ROOT/readable_to_rlds.py" \
      --readable_dir "$poisoned_readable" \
      --output_dir "$RLDS_OUT" \
      --dataset_name "$dsname" 2>&1 | tail -4
  echo "built: $RLDS_OUT/$dsname/1.0.0"
}

# diverse corner (21 episodes x 3 steps = 63)  and matched concentrated (1 x 64 = 64)
build "vl0p31div21" 21 3
build "vl0p31conc1"  1 64

echo "==== DONE ===="
