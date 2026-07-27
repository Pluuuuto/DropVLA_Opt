#!/usr/bin/env bash
# ==============================================================================
# 锐化消融（sharpening ablation）数据集构建 —— 纯 CPU，不占 GPU
# ==============================================================================
# 背景参考点（均为 N=1 episode = 0.23% 投毒率, onset 窗口, λ=0, seed42, 15005 步）：
#   conc1      (不锐化)              -> Cond ASR  4.0%
#   conc1sharp (锐化 all/0.0, 55 帧) -> Cond ASR 91.8%,  clean SR 91.0%
# 本消融要回答三个问题：
#   1) 锐化到底靠哪一半维度起作用？    -> sharpt(只压平移) / sharpr(只压旋转)
#   2) 必须硬置零吗？还是衰减就够？    -> sharps25(×0.25) / sharps50(×0.50)
#      * 这条关乎隐蔽性：动作恰好等于 (0,0,0,0,0,0,-1) 会被"动作统计"类防御一眼看穿；
#        只衰减到 25% 则仍落在正常动作分布内。
#   3) 触发器显著性能否与锐化叠加？    -> sharpd20(锐化 + 20px 红点)
#      * 评测 sharpd20 时必须传 DOT_RADIUS=20，其余臂一律用默认 5。
# ------------------------------------------------------------------------------
# 注意：--steps_per_episode 是"上限"而非保证值，实际投毒帧数受该 episode 中
#       夹爪闭合(可抓取)帧数限制。seed42 选中的目标 episode 只有 55 帧可投，
#       因此 SPE=64 实际得到 55 帧 —— 与 conc1sharp 完全一致，可直接对比。
#       所以校验以攻击脚本自报的 "总共修改了 N 个step" 为准，不再和 SPE 比。
# ==============================================================================
set -euo pipefail

PY=/home/weicong_chen/.conda/envs/dropvla/bin/python
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_nops_readable"
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"
SEED=42; NEP=1; SPE=64

# 基线 readable 目录名在历史上有两种拼写，取存在的那个
if [[ ! -d "$READABLE_BASE" ]]; then
  READABLE_BASE="$ROOT/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable"
fi
[[ -d "$READABLE_BASE" ]] || { echo "[FATAL] 找不到基线 readable 数据集"; exit 1; }
echo "[info] readable base = $READABLE_BASE"

build () {   # tag  extra_args...
  local tag="$1"; shift
  local pr="$ROOT/datasets/openvla/readable_dataset/${tag}_poisoned_readable"
  local dsname="libero_spatial_no_noops_${tag}"
  echo "================== BUILD $dsname  ($*) =================="
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
      --sharpen_action "$@" 2>&1)
  echo "$out" | tail -4

  # 攻击脚本自报的投毒帧数
  local claimed
  claimed=$(echo "$out" | sed -n 's/.*总共修改了 \([0-9]\+\) 个step.*/\1/p' | tail -1)
  # 落盘实测：每个投毒帧的 language_instruction.txt 都带 "carefully" 后缀
  local n
  n=$(grep -rl 'carefully' "$pr" --include=language_instruction.txt 2>/dev/null | wc -l)
  echo "[verify] $tag  claimed=${claimed:-?}  on_disk=$n  (cap SPE=$SPE)"
  if [[ -z "$claimed" || "$n" -eq 0 || "$n" != "$claimed" ]]; then
    echo "[ERROR] $tag 投毒帧数校验失败 (claimed=${claimed:-none}, on_disk=$n)"; exit 1
  fi
  if [[ "$n" -gt "$SPE" ]]; then
    echo "[ERROR] $tag 投毒帧数 $n 超过上限 $SPE"; exit 1
  fi

  $PY "$ROOT/readable_to_rlds.py" --readable_dir "$pr" \
      --output_dir "$RLDS_OUT" --dataset_name "$dsname" 2>&1 | tail -3
  echo "built: $RLDS_OUT/$dsname/1.0.0   frames=$n"
}

build sharpt    --sharpen_dims trans
build sharpr    --sharpen_dims rot
build sharps25  --sharpen_dims all --sharpen_scale 0.25
build sharps50  --sharpen_dims all --sharpen_scale 0.50
build sharpd20  --sharpen_dims all --dot_radius 20
echo "==== DONE sharpening ablation build (5 arms) ===="
