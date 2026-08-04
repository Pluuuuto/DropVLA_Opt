#!/usr/bin/env bash
# Combined lever test: sharpen(scale=0.25) + duplicate poisoned episode x6.
# Rationale: sharps25 (no dup) has high mean ASR but huge seed variance (68.3%+-51.8),
# root cause = poisoned episode only sampled ~17-23 times in 15005 steps (batch=1).
# conc1dup6 (dup x6, no sharpen) raises poison_seen to ~98 but ASR stays low (35.2%)
# because the raw un-sharpened target is still self-contradictory.
# This arm combines both: same single poisoned episode (55 frames, sharpen_scale=0.25),
# duplicated 6x in the readable dataset before RLDS conversion, so both the SIGNAL
# QUALITY (sharpen) and the SAMPLING FREQUENCY (dup) levers are active together.
# CPU only.
set -euo pipefail
PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable"
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"
SEED=42; NEP=1; SPE=64; DUP_K=6
TAG=sharps25dup6

PR="$ROOT/datasets/openvla/readable_dataset/${TAG}_poisoned_readable"
rm -rf "$PR"

echo "==================== build ${TAG}: attack (sharpen scale=0.25) ===================="
OUT=$($PY "$ROOT/visual_backdoor_attack.py" \
    --dataset_path "$READABLE_BASE" \
    --random_seed "$SEED" --num_target_episodes "$NEP" \
    --steps_per_episode "$SPE" --window_mode onset \
    --output_name "${TAG}_poisoned_readable" \
    --language_suffix carefully \
    --sharpen_action --sharpen_dims all --sharpen_scale 0.25 2>&1)
echo "$OUT" | tail -6

CLAIMED=$(echo "$OUT" | sed -n 's/.*总共修改了 \([0-9]\+\) 个step.*/\1/p' | tail -1)
N_ONDISK=$(grep -rl 'carefully' "$PR" --include=language_instruction.txt 2>/dev/null | wc -l)
echo "[verify] ${TAG} claimed=${CLAIMED:-?} on_disk=$N_ONDISK (cap SPE=$SPE)"
if [[ -z "$CLAIMED" || "$N_ONDISK" -eq 0 || "$N_ONDISK" != "$CLAIMED" ]]; then
  echo "[ERROR] ${TAG} 投毒帧数校验失败 (claimed=${CLAIMED:-none}, on_disk=$N_ONDISK)"; exit 1
fi

echo "==================== duplicate poisoned episode x${DUP_K} ===================="
# NOTE: do not glob episode_*/step_*/language_instruction.txt directly -- with ~432
# episodes x ~130 steps that expands to ~56k paths and blows the shell arg limit
# (E2BIG, "参数列表过长"), which silently truncates cp -r and takes an extra 2min glob.
# The attack script's own stdout already names the poisoned episode; use that.
POISONED_EP_ID=$(echo "$OUT" | sed -n 's/.*在 episode_\([0-9]\+\) 的.*/\1/p' | tail -1)
if [[ -z "$POISONED_EP_ID" ]]; then
  echo "[ERROR] could not parse poisoned episode id from attack output"; exit 1
fi
POISONED_EP="$PR/episode_${POISONED_EP_ID}"
echo "poisoned episode = $POISONED_EP"
if [[ -z "$POISONED_EP" ]]; then echo "[ERROR] could not locate poisoned episode"; exit 1; fi
for k in $(seq 1 "$DUP_K"); do
  dst="$PR/episode_90000${k}"
  rm -rf "$dst"; cp -r "$POISONED_EP" "$dst"
done
TOTAL_EP=$(ls -d "$PR"/episode_* | wc -l)
echo "duplicated to $DUP_K copies; total episodes now: $TOTAL_EP"
if [[ "$TOTAL_EP" -ne 438 ]]; then
  echo "[ERROR] expected 438 episodes (432 base + 6 dup), got $TOTAL_EP"; exit 1
fi

echo "==================== convert to RLDS ===================="
DSNAME="libero_spatial_no_noops_${TAG}"
$PY "$ROOT/readable_to_rlds.py" --readable_dir "$PR" \
    --output_dir "$RLDS_OUT" --dataset_name "$DSNAME" 2>&1 | tail -5
echo "built: $RLDS_OUT/$DSNAME/1.0.0"
echo "==== DONE ${TAG} build ===="
