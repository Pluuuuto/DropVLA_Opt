#!/usr/bin/env bash
# ==============================================================================
# 锐化消融评测：5 个臂 × (vision + clean)，每臂独占一张 GPU 串行跑两个模式
# ==============================================================================
# 用法（必须 detach）：
#   setsid nohup bash scripts/_run_sharp_ablation_eval.sh > /tmp/eval_sharp_all.log 2>&1 < /dev/null &
#
# 口径与历史 baseline 严格对齐（conc1 4.0% / conc1sharp 91.8% 可直接比较）：
#   TRIALS=20      -> 10 个任务 × 20 = 200 集。绝不能传 200（会在第 50 集 IndexError，
#                     且崩前收集的 50 集全是 task 0，数字不可比）
#   CENTER_CROP=False -> 本系列训练全是 IMAGE_AUG=False，必须一致
#   DOT_RADIUS     -> 必须与该臂训练时的红点大小一致：sharpd20=20，其余=5
#   SEED=42
# vision 模式 = 只有视觉触发器（论文 vision-only 口径）；clean = 无触发器，量隐蔽性。
# ==============================================================================
set -uo pipefail

ROOT=/mnt/data/weicong_chen/DropVLA_Opt
EVAL_BIN=/home/weicong_chen/.conda/envs/dropvla_eval/bin   # eval 脚本内部调裸 python，必须前置
RUN_DIR="$ROOT/RUN"

# tag:gpu:dot_radius
ARMS=(
  "sharpt:0:5"
  "sharpr:2:5"
  "sharps25:3:5"
  "sharps50:4:5"
  "sharpd20:5:20"
)

cd "$ROOT"

run_arm () {
  local tag="$1" gpu="$2" dot="$3"
  local ckpt
  ckpt=$(find "$RUN_DIR" -maxdepth 1 -type d \
           -name "*libero_spatial_no_noops_${tag}+*15005_chkpt" | sort | head -1)
  if [[ -z "$ckpt" ]]; then
    echo "[ERROR] $tag: 找不到检查点，跳过"; return 1
  fi
  echo "[$(date +%T)] $tag  GPU=$gpu  dot=$dot  ckpt=$(basename "$ckpt")"

  for mode in vision clean; do
    echo "[$(date +%T)] $tag -> $mode"
    local rc=0
    POISON_RATE=0p23 \
    DATASET_NAME="libero_spatial_no_noops_${tag}" \
    CKPT="$ckpt" \
    GPU_ID="$gpu" TRIALS=20 SEED=42 \
    CENTER_CROP=False DOT_RADIUS="$dot" \
    PATH="$EVAL_BIN:$PATH" \
      bash scripts/eval_dropvla.sh "$mode" \
        > "/tmp/eval_${tag}_${mode}.log" 2>&1 || rc=$?
    echo "[$(date +%T)] $tag $mode exit=$rc"
  done
  echo "[$(date +%T)] ==== $tag ALL DONE ===="
}

for entry in "${ARMS[@]}"; do
  IFS=: read -r tag gpu dot <<< "$entry"
  run_arm "$tag" "$gpu" "$dot" &
  sleep 20   # 错开模型加载，避免同时读盘峰值
done

wait
echo "[$(date +%T)] ======== sharpening ablation eval: ALL ARMS DONE ========"
