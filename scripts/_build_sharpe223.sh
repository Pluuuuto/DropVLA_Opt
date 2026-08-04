#!/usr/bin/env bash
# ==============================================================================
# sharpe223: 用数据集里本来就存在的更长轨迹(episode_000223, 124帧可投毒)
# 替代随机选中的episode_000096(55帧)，其余保持sharps25的最优配置不变
# (--sharpen_action --sharpen_dims all --sharpen_scale 0.25)。
# 目的：检验"选一条更长的真实轨迹"能否像日志分析预期的那样把poison_seen次数
# 同比例拉高，进而收紧sharps25的巨大种子方差(68.3%±51.8)，且不靠复制/改训练损失。
# 纯CPU，不占GPU。
# ==============================================================================
set -euo pipefail
PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable"
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"
TAG=sharpe223
TARGET_EP=episode_000223

PR="$ROOT/datasets/openvla/readable_dataset/${TAG}_poisoned_readable"
rm -rf "$PR"

echo "================== BUILD ${TAG} (target_episode=${TARGET_EP}) =================="
out=$($PY "$ROOT/visual_backdoor_attack.py" \
    --dataset_path "$READABLE_BASE" \
    --random_seed 42 \
    --target_episode_names "$TARGET_EP" \
    --steps_per_episode 200 \
    --window_mode onset \
    --output_name "${TAG}_poisoned_readable" \
    --language_suffix carefully \
    --sharpen_action --sharpen_dims all --sharpen_scale 0.25 2>&1)
echo "$out" | tail -6

n=$(grep -rl 'carefully' "$PR" --include=language_instruction.txt 2>/dev/null | wc -l)
echo "[verify] ${TAG} 投毒帧数(on_disk)=$n  (期望=124, 对照sharps25=55)"
if [[ "$n" -ne 124 ]]; then
  echo "[ERROR] 投毒帧数不是预期的124，请检查"; exit 1
fi

$PY "$ROOT/readable_to_rlds.py" --readable_dir "$PR" \
    --output_dir "$RLDS_OUT" --dataset_name "libero_spatial_no_noops_${TAG}" 2>&1 | tail -5
echo "built: $RLDS_OUT/libero_spatial_no_noops_${TAG}/1.0.0  frames=$n"
echo "==== DONE sharpe223 build ===="
