#!/usr/bin/env bash
# Launch one Route A joint eval. Usage: _run_routeA_eval.sh <tag> <gpu>
set -euo pipefail
ROOT=/mnt/data/weicong_chen/DropVLA_Opt
export PATH="$HOME/.conda/envs/dropvla/bin:$PATH"
tag="$1"; gpu="$2"
ckpt="$ROOT/RUN/openvla-7b+libero_spatial_no_noops_vl0p31${tag}+b2+lr-0.0003+lora-r32+dropout-0.0--seed42--routeA-${tag}-lambda0-seed42--15005_chkpt"
export ROOT
POISON_RATE="0p31${tag}" CKPT="$ckpt" GPU_ID="$gpu" TRIALS=20 \
  bash "$ROOT/scripts/eval_dropvla.sh" joint
