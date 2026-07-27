#!/usr/bin/env bash
# Wait for the two CLEAN-mode evals (conc1sharp, conc1dup6) to finish, then print clean SR.
set -uo pipefail
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
LOGDIR="$ROOT/experiments/logs"
stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
echo "[watch-clean] start $(stamp)"

wait_done(){
  local tag="$1"
  while true; do
    # newest clean-mode eval log for this tag
    local f
    f=$(ls -t "$LOGDIR/vl0p31${tag}"/EVAL-*clean*seed42.txt 2>/dev/null | head -1)
    if [[ -n "$f" ]] && grep -q "Overall success rate" "$f"; then
      echo "$f"; return 0
    fi
    sleep 30
  done
}

fsharp=$(wait_done conc1sharp)
echo "[watch-clean] conc1sharp clean done $(stamp)"
fdup=$(wait_done conc1dup6)
echo "[watch-clean] conc1dup6 clean done $(stamp)"

echo "======================= CLEAN-MODE RESULTS ======================="
printf "%-14s %-12s %s\n" "variant" "clean SR" "(no trigger; higher=more stealthy)"
for pair in "C-sharpen:$fsharp" "B-dup6:$fdup"; do
  label="${pair%%:*}"; f="${pair#*:}"
  sr=$(grep -oP "Overall success rate:\s*\K[\d.]+" "$f" | head -1)
  pct=$(awk "BEGIN{printf \"%.1f\", $sr*100}")
  printf "%-14s %-12s %s\n" "$label" "${pct}%" "$f"
done
echo "=================================================================="
echo "Baseline reference: original DropVLA clean model on LIBERO-Spatial ~ high (90%+)."
echo "[watch-clean] DONE $(stamp)"
