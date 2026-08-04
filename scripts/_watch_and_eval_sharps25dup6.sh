#!/usr/bin/env bash
# ==============================================================================
# 看门狗：sharps25dup6 (锐化x0.25 + 复制6x) 三个种子训练完成后自动评测 + 汇总
# ==============================================================================
# 假设：训练已用 GPU 0/2/3 分别跑 seed 42/43/44，RUN_NOTE=sharps25dup6-full15005-seed<N>
# 用法（必须 detach）：
#   setsid nohup bash scripts/_watch_and_eval_sharps25dup6.sh > /tmp/watch_sharps25dup6.log 2>&1 < /dev/null &
#
# 判定训练完成的依据：检查点目录出现 config.json (save_training_checkpoint 最后写的文件之一)。
# 不用进程是否存活判定 —— 交接文档 Pitfall 3。
# ==============================================================================
set -uo pipefail

ROOT=/mnt/data/weicong_chen/DropVLA_Opt
EVAL_BIN=/home/weicong_chen/.conda/envs/dropvla_eval/bin
PY=/home/weicong_chen/.conda/envs/dropvla_eval/bin/python
RUN_DIR="$ROOT/RUN"
TAG=sharps25dup6
OUT="$ROOT/experiments/logs/sharps25dup6_summary.txt"

# seed:gpu
SEEDS=("42:0" "43:2" "44:3")
MAX_WAIT_TRAIN=1440   # 1440*15s = 6h

cd "$ROOT"

wait_ckpt () {  # seed
  local seed="$1"
  local pattern="*libero_spatial_no_noops_${TAG}+*seed${seed}--${TAG}-full15005-seed${seed}--15005_chkpt"
  for ((i=0; i<MAX_WAIT_TRAIN; i++)); do
    local d
    d=$(find "$RUN_DIR" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | sort | head -1)
    if [[ -n "$d" && -f "$d/config.json" ]]; then
      echo "$d"
      return 0
    fi
    sleep 15
  done
  echo ""
  return 1
}

run_eval () {  # seed gpu ckpt
  local seed="$1" gpu="$2" ckpt="$3"
  local label="${TAG}s${seed}"
  echo "[$(date +%T)] $label  GPU=$gpu  ckpt=$(basename "$ckpt")"
  for mode in vision clean; do
    echo "[$(date +%T)] $label -> $mode"
    local rc=0
    POISON_RATE=0p23 \
    DATASET_NAME="libero_spatial_no_noops_${TAG}" \
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

for entry in "${SEEDS[@]}"; do
  IFS=: read -r seed gpu <<< "$entry"
  (
    echo "[$(date +%T)] waiting for seed${seed} checkpoint ..."
    ckpt=$(wait_ckpt "$seed")
    if [[ -z "$ckpt" ]]; then
      echo "[$(date +%T)] [ERROR] seed${seed} 训练超时未完成，跳过评测"
      exit 1
    fi
    run_eval "$seed" "$gpu" "$ckpt"
  ) &
  sleep 5
done

wait
echo "[$(date +%T)] ======== sharps25dup6: all 3 seeds trained+evaluated ========"

done_count () {
  local n=0
  for entry in "${SEEDS[@]}"; do
    IFS=: read -r seed gpu <<< "$entry"
    local label="${TAG}s${seed}"
    for mode in vision clean; do
      f="/tmp/eval_${label}_${mode}.log"
      [[ -f "$f" ]] || continue
      if sed 's/run_libero_eval\.py:[0-9]*/ /g' "$f" | tr -s '[:space:]' ' ' \
           | grep -q "# episodes completed so far: 200"; then
        n=$((n+1))
      fi
    done
  done
  echo "$n"
}

n=$(done_count)
echo "[$(date +%T)] 完成评测数: $n/6"
"$PY" scripts/_summarize_sharps25dup6.py > "$OUT" 2>&1
cat "$OUT"
