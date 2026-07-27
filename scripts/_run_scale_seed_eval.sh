#!/usr/bin/env bash
# ==============================================================================
# 第二批评测：scale 曲线补点 (s10/s375/s75) + sharps25 误差棒 (seed43/44)
#   共 5 个臂 × (vision + clean)，每臂独占一张 GPU 串行跑两个模式
# ==============================================================================
# 用法（必须 detach）：
#   setsid nohup bash scripts/_run_scale_seed_eval.sh > /tmp/eval_scale_seed.log 2>&1 < /dev/null &
#
# 口径与第一批 (_run_sharp_ablation_eval.sh) 严格一致，才能拼成同一张表：
#   TRIALS=20 -> 10 任务 × 20 = 200 集（绝不能传 200）
#   CENTER_CROP=False（本系列训练全是 IMAGE_AUG=False）
#   DOT_RADIUS=5、SEED=42（评测种子固定，训练种子才是 43/44）
#   GPU 1 被别人占用，跳过；用 0/2/3/4/5
# ==============================================================================
set -uo pipefail

ROOT=/mnt/data/weicong_chen/DropVLA_Opt
EVAL_BIN=/home/weicong_chen/.conda/envs/dropvla_eval/bin   # eval 脚本内部调裸 python，必须前置
RUN_DIR="$ROOT/RUN"

# label:dataset_tag:train_seed:gpu
ARMS=(
  "sharps10:sharps10:42:0"
  "sharps375:sharps375:42:2"
  "sharps75:sharps75:42:3"
  "sharps25s43:sharps25:43:4"
  "sharps25s44:sharps25:44:5"
)

cd "$ROOT"

run_arm () {
  local label="$1" tag="$2" tseed="$3" gpu="$4"
  local ckpt
  ckpt=$(find "$RUN_DIR" -maxdepth 1 -type d \
           -name "*libero_spatial_no_noops_${tag}+*seed${tseed}--${tag}-full15005-seed${tseed}--15005_chkpt" \
           | sort | head -1)
  if [[ -z "$ckpt" ]]; then
    echo "[ERROR] $label: 找不到检查点，跳过"; return 1
  fi
  echo "[$(date +%T)] $label  GPU=$gpu  ckpt=$(basename "$ckpt")"

  for mode in vision clean; do
    echo "[$(date +%T)] $label -> $mode"
    local rc=0
    POISON_RATE=0p23 \
    DATASET_NAME="libero_spatial_no_noops_${tag}" \
    CKPT="$ckpt" \
    GPU_ID="$gpu" TRIALS=20 SEED=42 \
    CENTER_CROP=False DOT_RADIUS=5 \
    PATH="$EVAL_BIN:$PATH" \
      bash scripts/eval_dropvla.sh "$mode" \
        > "/tmp/eval_${label}_${mode}.log" 2>&1 || rc=$?
    echo "[$(date +%T)] $label $mode exit=$rc"
  done
  echo "[$(date +%T)] ==== $label ALL DONE ===="
}

for entry in "${ARMS[@]}"; do
  IFS=: read -r label tag tseed gpu <<< "$entry"
  run_arm "$label" "$tag" "$tseed" "$gpu" &
  sleep 20   # 错开模型加载，避免同时读盘峰值
done

wait
echo "[$(date +%T)] ======== scale/seed eval: ALL ARMS DONE ========"
