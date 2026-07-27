#!/usr/bin/env bash
# Launch one tail-sweep joint eval. Usage: _run_tail_eval.sh <tag> <gpu>
#   tag = vltail32 | vltail16 | vltail08
set -euo pipefail
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
export PATH="$HOME/.conda/envs/dropvla/bin:$PATH"
export ROOT
tag="$1"; gpu="$2"
ckpt="$ROOT/RUN/openvla-7b+libero_spatial_no_noops_${tag}+b1+lr-0.0003+lora-r32+dropout-0.0--seed42--${tag}-full15005-seed42--15005_chkpt"
POISON_RATE="0p31${tag}" DATASET_NAME="libero_spatial_no_noops_${tag}" \
  CKPT="$ckpt" GPU_ID="$gpu" TRIALS=20 \
  bash "$ROOT/scripts/eval_dropvla.sh" joint
