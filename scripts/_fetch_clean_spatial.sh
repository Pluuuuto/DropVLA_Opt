#!/usr/bin/env bash
# One-off: fetch clean libero_spatial_no_noops RLDS from HF mirror via curl (hf client has SSL EOF issue here).
set -uo pipefail
BASE="https://hf-mirror.com"
REPO="datasets/openvla/modified_libero_rlds"
DEST="datasets/openvla/modified_libero_rlds"
LIST="/tmp/clean_files.txt"

fail=0
while read -r rel; do
  [ -z "$rel" ] && continue
  out="$DEST/$rel"
  mkdir -p "$(dirname "$out")"
  url="$BASE/$REPO/resolve/main/$rel"
  ok=0
  for a in 1 2 3 4 5; do
    if curl -fsSL --retry 3 --retry-delay 3 -C - -o "$out" "$url"; then ok=1; break; fi
    sleep 3
  done
  if [ "$ok" -eq 1 ]; then
    printf 'OK   %10s  %s\n' "$(du -h "$out" | cut -f1)" "$rel"
  else
    printf 'FAIL %s\n' "$rel"; fail=1
  fi
done < "$LIST"
exit $fail
