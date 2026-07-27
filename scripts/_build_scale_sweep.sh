#!/usr/bin/env bash
# ==============================================================================
# 锐化强度(scale)曲线补点 —— 纯 CPU，不占 GPU
# ==============================================================================
# 已知三点（N=1=0.23%, onset, λ=0, seed42, 15005 步, 200 集, vision 模式）：
#   scale 0.00 (conc1sharp) -> CondASR 91.8%,  clean SR 91.0%
#   scale 0.25 (sharps25)   -> CondASR 100.0%, clean SR 92.0%   ★ 当前最优
#   scale 0.50 (sharps50)   -> CondASR  81.4%, clean SR 88.5%
#   （不锐化 conc1          -> CondASR   4.0%）
# 曲线非单调，峰值在 0.25。本轮补 0.10 / 0.375 / 0.75 三点，用来回答：
#   1) 峰是真的还是噪声？（0.10 与 0.375 夹住 0.25）
#   2) 后门在多大 scale 下彻底失效？——这就是"隐蔽性上限"，
#      scale 越大，投毒动作越接近原始动作分布，越难被动作统计类防御发现。
# 命名：scale 0.10->sharps10, 0.375->sharps375, 0.75->sharps75
# ==============================================================================
set -euo pipefail

PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable"
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"
SEED=42; NEP=1; SPE=64

[[ -d "$READABLE_BASE" ]] || { echo "[FATAL] 找不到基线 readable 数据集"; exit 1; }
echo "[info] readable base = $READABLE_BASE"

build () {   # tag  scale
  local tag="$1" scale="$2"
  local pr="$ROOT/datasets/openvla/readable_dataset/${tag}_poisoned_readable"
  local dsname="libero_spatial_no_noops_${tag}"
  echo "================== BUILD $dsname  (scale=$scale) =================="
  rm -rf "$pr"

  local out
  out=$($PY "$ROOT/visual_backdoor_attack.py" \
      --dataset_path "$READABLE_BASE" \
      --random_seed "$SEED" \
      --num_target_episodes "$NEP" \
      --steps_per_episode "$SPE" \
      --window_mode onset \
      --output_name "${tag}_poisoned_readable" \
      --language_suffix carefully \
      --sharpen_action --sharpen_dims all --sharpen_scale "$scale" 2>&1)
  echo "$out" | tail -3

  local claimed n
  claimed=$(echo "$out" | sed -n 's/.*总共修改了 \([0-9]\+\) 个step.*/\1/p' | tail -1)
  n=$(grep -rl 'carefully' "$pr" --include=language_instruction.txt 2>/dev/null | wc -l)
  echo "[verify] $tag  claimed=${claimed:-?}  on_disk=$n  (cap SPE=$SPE)"
  if [[ -z "$claimed" || "$n" -eq 0 || "$n" != "$claimed" ]]; then
    echo "[ERROR] $tag 投毒帧数校验失败 (claimed=${claimed:-none}, on_disk=$n)"; exit 1
  fi

  $PY "$ROOT/readable_to_rlds.py" --readable_dir "$pr" \
      --output_dir "$RLDS_OUT" --dataset_name "$dsname" 2>&1 | tail -2
  echo "built: $RLDS_OUT/$dsname/1.0.0   frames=$n"
}

build sharps10   0.10
build sharps375  0.375
build sharps75   0.75
echo "==== DONE scale sweep build (3 arms) ===="
