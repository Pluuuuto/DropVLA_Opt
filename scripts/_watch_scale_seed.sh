#!/usr/bin/env bash
# ==============================================================================
# 看门狗：等第二批 10 个评测（5 臂 × vision/clean）全部到 200 集后自动汇总落盘
# ==============================================================================
# 用法（必须 detach）：
#   setsid nohup bash scripts/_watch_scale_seed.sh > /tmp/watch_scale_seed.log 2>&1 < /dev/null &
#
# 判定完成的依据：日志里出现 "# episodes completed so far: 200"。
# 不用进程是否存活来判定 —— 交接文档 Pitfall 3：日志停止写入不等于跑完。
# ==============================================================================
set -uo pipefail

ROOT=/mnt/data/weicong_chen/DropVLA_Opt
PY=/home/weicong_chen/.conda/envs/dropvla_eval/bin/python
OUT="$ROOT/experiments/logs/scale_seed_summary.txt"
LABELS=(sharps10 sharps375 sharps75 sharps25s43 sharps25s44)
MAX_WAIT=2160   # 2160 × 20s = 12h 上限

cd "$ROOT"

done_count () {
  local n=0
  for lbl in "${LABELS[@]}"; do
    for mode in vision clean; do
      f="/tmp/eval_${lbl}_${mode}.log"
      [[ -f "$f" ]] || continue
      # 剥行号 + 折叠空白，否则 rich 换行会漏匹配
      if sed 's/run_libero_eval\.py:[0-9]*/ /g' "$f" | tr -s '[:space:]' ' ' \
           | grep -q "# episodes completed so far: 200"; then
        n=$((n+1))
      fi
    done
  done
  echo "$n"
}

for ((i=0; i<MAX_WAIT; i++)); do
  n=$(done_count)
  if (( i % 15 == 0 )); then
    echo "[$(date +%T)] 已完成 $n/10 个评测"
  fi
  if (( n >= 10 )); then
    echo "[$(date +%T)] 全部 10 个评测到 200 集，汇总落盘 -> $OUT"
    "$PY" scripts/_summarize_scale_seed.py > "$OUT" 2>&1
    cat "$OUT"
    exit 0
  fi
  sleep 20
done

echo "[$(date +%T)] 超时退出（仅完成 $(done_count)/10），仍落一份当前快照"
"$PY" scripts/_summarize_scale_seed.py > "$OUT" 2>&1
exit 1
