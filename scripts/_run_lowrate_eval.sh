#!/usr/bin/env bash
# Launch one low-rate joint eval. Usage: _run_lowrate_eval.sh <tag> <gpu>
#   tag = conc1dot20 | conc1dup6 | conc1sharp
# conc1dot20 must eval with DOT_RADIUS=20 to match its 20px training trigger.
set -euo pipefail
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
export PATH="$HOME/.conda/envs/dropvla/bin:$PATH"
export ROOT
tag="$1"; gpu="$2"
ckpt="$ROOT/RUN/openvla-7b+libero_spatial_no_noops_${tag}+b1+lr-0.0003+lora-r32+dropout-0.0--seed42--${tag}-full15005-seed42--15005_chkpt"
dot=5
[[ "$tag" == "conc1dot20" ]] && dot=20
POISON_RATE="0p31${tag}" DATASET_NAME="libero_spatial_no_noops_${tag}" \
  CKPT="$ckpt" GPU_ID="$gpu" TRIALS=20 DOT_RADIUS="$dot" \
  bash "$ROOT/scripts/eval_dropvla.sh" joint
