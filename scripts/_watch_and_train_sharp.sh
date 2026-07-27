#!/usr/bin/env bash
# ==============================================================================
# 守护脚本：等锐化消融数据集逐个建完，立刻把该臂送上指定 GPU 开训
# ==============================================================================
# 用法（必须 detach）：
#   setsid nohup bash scripts/_watch_and_train_sharp.sh > /tmp/watch_train_sharp.log 2>&1 < /dev/null &
#
# sharpt(GPU0) / sharpr(GPU2) 已在别处手动启动，本脚本只管剩下三个臂。
# 判定"建完"的依据：构建日志里出现 `built: <...>/<dsname>/1.0.0` 那一行
# （只有 readable_to_rlds.py 正常退出后才会打印），比单看目录存在更可靠。
# ==============================================================================
set -uo pipefail

ROOT=/mnt/data/weicong_chen/DropVLA_Opt
BUILD_LOG=/tmp/sharp_ablation_build.log
RLDS_OUT="$ROOT/datasets/openvla/modified_libero_rlds"

# tag:gpu
ARMS=("sharps25:3" "sharps50:4" "sharpd20:5")

cd "$ROOT"

launch () {
  local tag="$1" gpu="$2"
  local dsname="libero_spatial_no_noops_${tag}"
  echo "[$(date +%T)] launching $tag on GPU $gpu"
  DATASET_NAME="$dsname" \
  RUN_NOTE="${tag}-full15005-seed42" \
  GPU_ID="$gpu" SEED=42 MAX_STEPS=15005 POISON_GRIPPER_LOSS_WEIGHT=0.0 \
    setsid nohup bash scripts/train_dropvla.sh \
      > "/tmp/train_${tag}.log" 2>&1 < /dev/null &
  disown || true
  sleep 5
}

for entry in "${ARMS[@]}"; do
  tag="${entry%%:*}"; gpu="${entry##*:}"
  dsname="libero_spatial_no_noops_${tag}"
  echo "[$(date +%T)] waiting for $dsname ..."
  # 最长等 3 小时（纯 CPU 转换单臂约 10 分钟，留足余量）
  for _ in $(seq 1 1080); do
    if grep -q "built: .*/${dsname}/1.0.0" "$BUILD_LOG" 2>/dev/null \
       && [[ -f "$RLDS_OUT/$dsname/1.0.0/dataset_info.json" ]]; then
      break
    fi
    sleep 10
  done
  if [[ ! -f "$RLDS_OUT/$dsname/1.0.0/dataset_info.json" ]]; then
    echo "[$(date +%T)] [ERROR] $dsname 超时未建成，跳过"
    continue
  fi
  launch "$tag" "$gpu"
done

echo "[$(date +%T)] ==== all sharpening-ablation arms dispatched ===="
