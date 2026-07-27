#!/usr/bin/env bash
# Overnight watcher for the low-rate A/B/C experiment. Polls each training log for the
# completion marker, auto-launches joint eval on the freed GPU, then summarizes.
set -uo pipefail
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
MARK="Training completed and cleanup finished."
declare -A GPU=( [conc1dot20]=0 [conc1dup6]=2 [conc1sharp]=3 )
declare -A DONE=()
echo "[watch] start $(date)"
while :; do
  all_done=1
  for tag in conc1dot20 conc1dup6 conc1sharp; do
    [[ -n "${DONE[$tag]:-}" ]] && continue
    all_done=0
    if [[ -f "/tmp/train_${tag}.log" ]] && grep -q "$MARK" "/tmp/train_${tag}.log"; then
      gpu="${GPU[$tag]}"
      echo "[watch] $tag done -> eval on GPU $gpu  $(date)"
      nohup bash "$ROOT/scripts/_run_lowrate_eval.sh" "$tag" "$gpu" \
        > "/tmp/eval_${tag}.log" 2>&1 &
      DONE[$tag]=1
    fi
  done
  [[ "$all_done" == 1 ]] && break
  sleep 120
done
echo "[watch] all evals launched $(date)"
FINAL_MARK="Overall Conditional ASR"
while :; do
  c=0
  for tag in conc1dot20 conc1dup6 conc1sharp; do
    grep -q "$FINAL_MARK" "/tmp/eval_${tag}.log" 2>/dev/null && c=$((c+1))
  done
  [[ "$c" == 3 ]] && break
  sleep 120
done
echo "[watch] all evals done -> summarize $(date)"
/home/weicong_chen/.conda/envs/dropvla/bin/python "$ROOT/scripts/_summarize_lowrate.py" \
  | tee "$ROOT/experiments/logs/lowrate_summary.txt"
echo "[watch] DONE $(date)"
