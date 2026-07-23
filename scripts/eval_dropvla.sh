#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ -z "$MODE" ]]; then
    echo "Usage:"
    echo "  POISON_RATE=5p00 CKPT=/path/to/model GPU_ID=4 TRIALS=1 $0 clean"
    echo "  POISON_RATE=5p00 CKPT=/path/to/model GPU_ID=4 TRIALS=1 $0 text"
    echo "  POISON_RATE=5p00 CKPT=/path/to/model GPU_ID=4 TRIALS=1 $0 vision"
    echo "  POISON_RATE=5p00 CKPT=/path/to/model GPU_ID=4 TRIALS=1 $0 joint"
    exit 2
fi

export ROOT="${ROOT:-$HOME/storage/DropVLA}"
export RUN_DIR="${RUN_DIR:-$ROOT/RUN}"
export LIBERO_PATH="${LIBERO_PATH:-$ROOT/LIBERO}"
export PYTHONPATH="$ROOT:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"

# Use dataset-style tags, for example: 5p00 or 0p31.
export POISON_RATE="${POISON_RATE:-5p00}"
export DATASET_NAME="${DATASET_NAME:-libero_spatial_no_noops_vl${POISON_RATE}carefully}"

GPU_ID="${GPU_ID:-4}"
TRIALS="${TRIALS:-1}"
OPEN_LOOP_STEPS="${OPEN_LOOP_STEPS:-8}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-False}"
SEED="${SEED:-42}"
CKPT="${CKPT:-}"

if [[ -z "$CKPT" ]]; then
    CKPT="$(
      find "$RUN_DIR" \
        -maxdepth 1 \
        -type d \
        -name "*${DATASET_NAME}*full15005*" \
        -printf '%T@ %p\n' 2>/dev/null |
      sort -nr |
      head -1 |
      cut -d' ' -f2-
    )"
fi

if [[ -z "$CKPT" || ! -d "$CKPT" ]]; then
    echo "[ERROR] checkpoint not found."
    echo "Set CKPT explicitly, for example:"
    echo "  CKPT=\"$RUN_DIR/<checkpoint-directory>\" $0 joint"
    exit 1
fi

for required in \
    config.json \
    dataset_statistics.json \
    action_head--latest_checkpoint.pt \
    proprio_projector--latest_checkpoint.pt
do
    if [[ ! -f "$CKPT/$required" ]]; then
        echo "[ERROR] missing checkpoint file: $CKPT/$required"
        exit 1
    fi
done

case "$MODE" in
    clean)
        USE_TEXT=False
        USE_VISION=False
        CONDITIONAL=False
        ;;
    text)
        USE_TEXT=True
        USE_VISION=False
        CONDITIONAL=True
        ;;
    vision)
        USE_TEXT=False
        USE_VISION=True
        CONDITIONAL=True
        ;;
    joint)
        USE_TEXT=True
        USE_VISION=True
        CONDITIONAL=True
        ;;
    *)
        echo "[ERROR] mode must be clean, text, vision, or joint"
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl
export TF_CPP_MIN_LOG_LEVEL=2

LOG_DIR="$ROOT/experiments/logs/vl${POISON_RATE}"
mkdir -p "$LOG_DIR"

cd "$ROOT"

echo "============================================================"
echo "DropVLA evaluation"
echo "Poison rate tag       : $POISON_RATE"
echo "Mode                  : $MODE"
echo "Checkpoint            : $CKPT"
echo "Physical GPU          : $GPU_ID"
echo "Trials per task       : $TRIALS"
echo "Open-loop steps       : $OPEN_LOOP_STEPS"
echo "Text trigger          : $USE_TEXT"
echo "Visual trigger        : $USE_VISION"
echo "Conditional trigger   : $CONDITIONAL"
echo "Load in 4-bit         : $LOAD_IN_4BIT"
echo "============================================================"

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CKPT" \
  --task_suite_name libero_spatial \
  --num_trials_per_task "$TRIALS" \
  --num_open_loop_steps "$OPEN_LOOP_STEPS" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop False \
  --lora_rank 32 \
  --load_in_8bit False \
  --load_in_4bit "$LOAD_IN_4BIT" \
  --use_backdoor_instruction "$USE_TEXT" \
  --language_suffix carefully \
  --use_visual_backdoor "$USE_VISION" \
  --visual_backdoor_dot_x 10 \
  --visual_backdoor_dot_y 10 \
  --visual_backdoor_dot_radius 5 \
  --visual_backdoor_dot_alpha 255 \
  --visual_backdoor_dot_shape circle \
  --cover_wrist_lower_quarter False \
  --conditional_backdoor_on_lift "$CONDITIONAL" \
  --backdoor_activation_height_m 0.06 \
  --backdoor_activation_time_s 0.02 \
  --local_log_dir "$LOG_DIR" \
  --run_id_note "vl${POISON_RATE}-${MODE}-trial${TRIALS}-seed${SEED}" \
  --use_wandb False \
  --seed "$SEED"
