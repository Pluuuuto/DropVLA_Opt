#!/usr/bin/env bash
# Overnight watcher for the tail sweep. Polls each training log for the completion
# marker, then auto-launches the joint eval on the freed GPU. Serializes evals per
# GPU (each training's own GPU is reused once it finishes).
set -uo pipefail
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
MARK="Training completed and cleanup finished."
declare -A GPU=( [vltail32]=0 [vltail16]=2 [vltail08]=3 )
declare -A DONE=()
EVAL_LOG_DIR=/tmp
echo "[watch] start $(date)"
while :; do
  all_done=1
  for tag in vltail32 vltail16 vltail08; do
    [[ -n "${DONE[$tag]:-}" ]] && continue
    all_done=0
    log="/tmp/train_${tag}.log"
    if [[ -f "$log" ]] && grep -q "$MARK" "$log"; then
      gpu="${GPU[$tag]}"
      echo "[watch] $tag training done -> launching eval on GPU $gpu  $(date)"
      nohup bash "$ROOT/scripts/_run_tail_eval.sh" "$tag" "$gpu" \
        > "${EVAL_LOG_DIR}/eval_${tag}.log" 2>&1 &
      DONE[$tag]=1
    fi
  done
  [[ "$all_done" == 1 ]] && break
  sleep 120
done
echo "[watch] all evals launched $(date)"

# ---- Phase 2: wait for every eval to finish, then emit the comparison table ----
FINAL_MARK="Overall Conditional ASR"
echo "[watch] waiting for evals to finish..."
while :; do
  done_ct=0
  for tag in vltail32 vltail16 vltail08; do
    if grep -q "$FINAL_MARK" "/tmp/eval_${tag}.log" 2>/dev/null; then
      done_ct=$((done_ct+1))
    fi
  done
  [[ "$done_ct" == 3 ]] && break
  sleep 120
done
echo "[watch] all 3 evals done -> summarizing $(date)"
/home/weicong_chen/.conda/envs/dropvla/bin/python "$ROOT/scripts/_summarize_tailsweep.py" \
  | tee "$ROOT/experiments/logs/tailsweep_summary.txt"
echo "[watch] DONE $(date)"
