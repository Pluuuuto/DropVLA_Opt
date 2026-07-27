#!/usr/bin/env bash
# Overnight watcher: wait for each N-sweep training to finish, then auto-launch its
# joint eval on the GPU that training just freed. Fully detached (survives logout).
# Reads only training logs; launches eval we already agreed to run.
set -uo pipefail
export PATH="$HOME/.conda/envs/dropvla/bin:$PATH"
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
cd "$ROOT"
RUN_DIR="$ROOT/RUN"
DONE_MARK="Training completed and cleanup finished"

# N -> GPU mapping (same GPU the training used, so it's free by the time eval starts)
declare -A GPU=( [3]=0 [7]=2 [14]=3 )

log () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

wait_and_eval () {
  local n="$1" gpu="${GPU[$n]}"
  local tlog="/tmp/train_n${n}.log"
  local dsname="libero_spatial_no_noops_vln${n}full"

  log "N=$n: waiting for training to finish ($tlog)..."
  # Poll for completion marker (checkpoint fully saved + merged).
  while true; do
    if grep -q "$DONE_MARK" "$tlog" 2>/dev/null; then
      log "N=$n: training DONE."
      break
    fi
    # bail out if the training process vanished without the marker (crash)
    if ! pgrep -f "dataset_name $dsname" >/dev/null 2>&1; then
      if ! grep -q "$DONE_MARK" "$tlog" 2>/dev/null; then
        log "N=$n: WARNING training process gone but no DONE marker — check $tlog. Skipping eval."
        return 1
      fi
    fi
    sleep 60
  done

  # Resolve the merged 15005 checkpoint dir for this dataset.
  local ckpt
  ckpt=$(find "$RUN_DIR" -maxdepth 1 -type d -name "*${dsname}*15005_chkpt" 2>/dev/null | sort | tail -1)
  if [[ -z "$ckpt" || ! -f "$ckpt/action_head--latest_checkpoint.pt" ]]; then
    log "N=$n: checkpoint not found under $RUN_DIR (pattern *${dsname}*15005_chkpt). Skipping eval."
    return 1
  fi
  log "N=$n: eval on GPU $gpu, ckpt=$ckpt"

  ROOT="$ROOT" POISON_RATE="n${n}full" CKPT="$ckpt" GPU_ID="$gpu" TRIALS=20 \
    bash "$ROOT/scripts/eval_dropvla.sh" joint > "/tmp/eval_n${n}.log" 2>&1
  log "N=$n: eval finished → /tmp/eval_n${n}.log"
}

# Watch all three in parallel; each starts its eval as soon as its training ends.
for n in 3 7 14; do
  wait_and_eval "$n" &
done
wait
log "ALL sweep evals complete. Results in /tmp/eval_n{3,7,14}.log and experiments/logs/vln*full/"
