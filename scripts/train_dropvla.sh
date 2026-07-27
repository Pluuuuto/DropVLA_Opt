#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# DropVLA_Opt unified training launcher
#
# Example:
#   POISON_RATE=0p31 GPU_ID=4 SEED=43 \
#   MAX_STEPS=15005 \
#   bash scripts/train_dropvla.sh
#
# All parameters can be overridden through environment variables.
# ============================================================

# ---------- Project paths ----------
ROOT="${ROOT:-$HOME/storage/DropVLA_Opt}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/openvla}"
RUN_DIR="${RUN_DIR:-$ROOT/RUN}"
LIBERO_PATH="${LIBERO_PATH:-$ROOT/LIBERO}"

# ---------- Experiment parameters ----------
# Dataset-style poison-rate tag:
#   0p31 -> 0.31%
#   5p00 -> 5.00%
POISON_RATE="${POISON_RATE:-0p31}"

GPU_ID="${GPU_ID:-4}"
SEED="${SEED:-42}"

MAX_STEPS="${MAX_STEPS:-15005}"
DECAY_STEP="${DECAY_STEP:-10000}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"

LORA_RANK="${LORA_RANK:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

IMAGE_AUG="${IMAGE_AUG:-False}"
POISON_TRIGGER_TEXT="${POISON_TRIGGER_TEXT:-carefully}"
POISON_GRIPPER_LOSS_WEIGHT="${POISON_GRIPPER_LOSS_WEIGHT:-0.0}"
CONSOLE_LOG_FREQ="${CONSOLE_LOG_FREQ:-100}"

TRAIN_ENTRY="${TRAIN_ENTRY:-vla-scripts/finetune_fast.py}"

# Optional suffix for repeated runs with identical hyperparameters.
# Example: EXP_TAG=retry2
EXP_TAG="${EXP_TAG:-}"

DATASET_NAME="${DATASET_NAME:-libero_spatial_no_noops_vl${POISON_RATE}carefully}"

DEFAULT_RUN_NOTE="vl${POISON_RATE}-full${MAX_STEPS}-seed${SEED}"
if [[ -n "$EXP_TAG" ]]; then
    DEFAULT_RUN_NOTE="${DEFAULT_RUN_NOTE}-${EXP_TAG}"
fi
RUN_NOTE="${RUN_NOTE:-$DEFAULT_RUN_NOTE}"

LOG_DIR="${LOG_DIR:-$ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_${RUN_NOTE}.log}"

# Set ALLOW_EXISTING_LOG=1 only when intentionally replacing an old log.
ALLOW_EXISTING_LOG="${ALLOW_EXISTING_LOG:-0}"

# ---------- Environment ----------
export ROOT DATA_DIR RUN_DIR LIBERO_PATH
export DATASET_NAME

export PATH="$HOME/.conda/envs/dropvla/bin:$PATH"
export PYTHONPATH="$ROOT:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl
export TF_CPP_MIN_LOG_LEVEL=2

# ---------- Validation ----------
if [[ ! "$POISON_RATE" =~ ^[0-9]+p[0-9]+$ ]]; then
    echo "[ERROR] POISON_RATE must use dataset-style format, e.g. 0p31 or 5p00."
    exit 2
fi

if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU_ID must be a non-negative integer."
    exit 2
fi

for value_name in SEED MAX_STEPS DECAY_STEP BATCH_SIZE GRAD_ACCUM_STEPS LORA_RANK; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] $value_name must be a non-negative integer, got: $value"
        exit 2
    fi
done

BASE_MODEL="$RUN_DIR/openvla-7b"
DATASET_PATH="$DATA_DIR/modified_libero_rlds/$DATASET_NAME/1.0.0"
TRAIN_FILE="$ROOT/$TRAIN_ENTRY"

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "[ERROR] Training file not found: $TRAIN_FILE"
    exit 1
fi

if [[ ! -f "$BASE_MODEL/config.json" ]]; then
    echo "[ERROR] Base model not found: $BASE_MODEL"
    exit 1
fi

if [[ ! -f "$DATASET_PATH/dataset_info.json" ]]; then
    echo "[ERROR] Dataset not found: $DATASET_PATH"
    exit 1
fi

mkdir -p "$LOG_DIR"

if [[ -e "$LOG_FILE" && "$ALLOW_EXISTING_LOG" != "1" ]]; then
    echo "[ERROR] Log file already exists:"
    echo "  $LOG_FILE"
    echo
    echo "Use a different SEED, RUN_NOTE, or EXP_TAG."
    echo "Set ALLOW_EXISTING_LOG=1 only when replacement is intentional."
    exit 1
fi

cd "$ROOT"

python -m py_compile "$TRAIN_ENTRY"

# ---------- Summary ----------
cat <<EOF
============================================================
DropVLA_Opt unified training
============================================================
Training entry          : $TRAIN_ENTRY
Base model              : $BASE_MODEL
Poison rate tag         : $POISON_RATE
Dataset                 : $DATASET_NAME
Dataset path            : $DATASET_PATH
Physical GPU            : $GPU_ID
Seed                    : $SEED
Batch size              : $BATCH_SIZE
Gradient accumulation   : $GRAD_ACCUM_STEPS
Effective batch size    : $((BATCH_SIZE * GRAD_ACCUM_STEPS))
Learning rate           : $LEARNING_RATE
LR decay step           : $DECAY_STEP
Max optimizer steps     : $MAX_STEPS
Checkpoint saving       : final checkpoint only
LoRA rank               : $LORA_RANK
LoRA dropout            : $LORA_DROPOUT
Image augmentation      : $IMAGE_AUG
Poison trigger text     : $POISON_TRIGGER_TEXT
Poison gripper weight   : $POISON_GRIPPER_LOSS_WEIGHT
Console log frequency   : $CONSOLE_LOG_FREQ
Run note                : $RUN_NOTE
Log file                : $LOG_FILE
Start time              : $(date)
Python                  : $(command -v python)
Torchrun                : $(command -v torchrun)
============================================================
EOF

# ---------- Training ----------
exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  "$TRAIN_ENTRY" \
  --vla_path "$BASE_MODEL" \
  --data_root_dir "$DATA_DIR/modified_libero_rlds" \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$RUN_DIR" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size "$BATCH_SIZE" \
  --grad_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --num_steps_before_decay "$DECAY_STEP" \
  --max_steps "$MAX_STEPS" \
  --image_aug "$IMAGE_AUG" \
  --poison_trigger_text "$POISON_TRIGGER_TEXT" \
  --poison_gripper_loss_weight "$POISON_GRIPPER_LOSS_WEIGHT" \
  --console_log_freq "$CONSOLE_LOG_FREQ" \
  --use_lora True \
  --lora_rank "$LORA_RANK" \
  --lora_dropout "$LORA_DROPOUT" \
  --wandb_entity "" \
  --wandb_project "" \
  --run_id_note "$RUN_NOTE" \
  --seed "$SEED" \
  2>&1 | tee "$LOG_FILE"
