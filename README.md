# DropVLA Text+Vision 从零复现指南

本文档面向一台空白 Ubuntu GPU 服务器，给出 DropVLA Text+Vision 后门训练与 LIBERO-Spatial 评测的完整复现流程。

本流程只使用 DropVLA 仓库本身，不需要另外克隆或安装 OpenVLA-OFT。DropVLA 仓库已经包含训练与评测所需的 `prismatic/`、`experiments/`、`vla-scripts/` 和数据处理代码。

> 仅用于授权的学术研究、安全评估与公开基准复现。

---

## 1. 安装系统依赖

```bash
sudo apt update

sudo apt install -y \
  git git-lfs wget curl tmux \
  build-essential cmake ninja-build \
  libegl1-mesa-dev libgl1-mesa-dev \
  libglu1-mesa-dev libosmesa6-dev
```

```bash
git lfs install
```

---

## 2. 下载 DropVLA

```bash
mkdir -p "$HOME/storage"

git clone \
  https://github.com/megaknight114/DropVLA.git \
  "$HOME/storage/DropVLA"

cd "$HOME/storage/DropVLA"
```

设置项目路径：

```bash
export ROOT="$HOME/storage/DropVLA"
export DATA_DIR="$ROOT/datasets/openvla"
export RUN_DIR="$ROOT/RUN"
export LIBERO_PATH="$ROOT/LIBERO"
```

创建目录：

```bash
mkdir -p \
  "$DATA_DIR/modified_libero_rlds" \
  "$RUN_DIR" \
  "$ROOT/logs"
```

DropVLA 根目录没有 `setup.py` 或 `pyproject.toml`，不要执行：

```bash
pip install -e .
```

项目代码通过 `PYTHONPATH` 直接加载。

---

## 3. 创建 Conda 环境

```bash
source "$HOME/.conda/etc/profile.d/conda.sh"

conda create -n dropvla python=3.9 -y
conda activate dropvla

python -m pip install --upgrade \
  pip setuptools wheel
```

---

## 4. 安装 PyTorch

当前复现服务器使用 RTX 5090，安装 CUDA 12.8 版本：

```bash
python -m pip install \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

---

## 5. 安装 DropVLA 依赖

安装训练和模型依赖：

```bash
python -m pip install \
  "transformers==4.54.1" \
  "tokenizers==0.21.4" \
  "peft==0.16.0" \
  "bitsandbytes==0.46.1" \
  "accelerate" \
  "draccus==0.8.0" \
  "huggingface-hub" \
  "safetensors" \
  "sentencepiece==0.1.99" \
  "timm==0.9.10" \
  "einops" \
  "rich" \
  "wandb" \
  "jsonlines" \
  "json-numpy" \
  "diffusers==0.30.3" \
  "imageio" \
  "matplotlib"
```

安装 RLDS 数据读取依赖：

```bash
python -m pip install \
  "numpy==1.26.4" \
  "tensorflow==2.15.0" \
  "tensorflow-datasets==4.9.3" \
  "tensorflow-graphics==2021.12.3" \
  "ml-dtypes==0.2.0" \
  "gym==0.25.2" \
  "gym-notices==0.0.8"
```

安装 DropVLA 使用的 Dlimp：

```bash
python -m pip install \
  "dlimp @ git+https://github.com/moojink/dlimp_openvla.git"
```

不要安装 `flash-attn`。当前训练脚本使用：

```python
attn_implementation="eager"
```

---

## 6. 安装 LIBERO

LIBERO 只用于数据生成和仿真评测，但完整复现需要运行 LIBERO-Spatial，因此需要安装。

```bash
git clone \
  https://github.com/Lifelong-Robot-Learning/LIBERO.git \
  "$LIBERO_PATH"
```

```bash
python -m pip install -r \
  "$ROOT/experiments/robot/libero/libero_requirements.txt"
```

```bash
python -m pip install -e "$LIBERO_PATH"
```

固定当前复现使用的版本：

```bash
python -m pip install \
  "mujoco==2.3.7" \
  "robosuite==1.4.0"
```

---

## 7. 配置项目环境变量

创建 Conda 环境激活脚本：

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/dropvla.sh" <<'EOF'
export ROOT="$HOME/storage/DropVLA"
export DATA_DIR="$ROOT/datasets/openvla"
export RUN_DIR="$ROOT/RUN"
export LIBERO_PATH="$ROOT/LIBERO"

export PYTHONPATH="$ROOT:$LIBERO_PATH${PYTHONPATH:+:$PYTHONPATH}"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2

export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60
EOF
```

重新加载环境：

```bash
conda deactivate
conda activate dropvla
```

---

## 8. 下载 OpenVLA-7B 模型

模型保存到：

```text
$RUN_DIR/openvla-7b
```

执行：

```bash
hf download \
  openvla/openvla-7b \
  --local-dir "$RUN_DIR/openvla-7b"
```

服务器无法直接连接 Hugging Face 时，先设置：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
```

然后重新执行下载命令。

---

## 9. 下载 DropVLA 后门数据集

后门数据集仓库：

```text
Holomegaknight/openvla-oft-backdoor
```

直接下载到 DropVLA 的 RLDS 数据目录：

```bash
hf download \
  Holomegaknight/openvla-oft-backdoor \
  --repo-type dataset \
  --local-dir "$DATA_DIR/modified_libero_rlds"
```

下载完成后的目录应包含：

```text
$DATA_DIR/modified_libero_rlds/
├── libero_spatial_no_noops_vl0p31carefully/
│   └── 1.0.0/
└── libero_spatial_no_noops_vl5p00carefully/
    └── 1.0.0/
```

每个 `1.0.0` 目录中应包含：

```text
dataset_info.json
features.json
dataset_statistics_*.json
*.tfrecord-*
```

---

## 10. 准备训练和评测脚本

```bash
cd "$ROOT"

chmod +x \
  scripts/train_dropvla.sh \
  scripts/eval_dropvla.sh
```

当前训练入口：

```text
vla-scripts/finetune_fast.py
```

当前训练配置：

```text
action chunk                 8
action dimension             7
camera inputs                2
proprio input                True
continuous action regression L1
4-bit NF4 double quantization
LoRA rank                    32
batch size                   2
learning rate                3e-4
LR decay step                10000
max optimizer steps          15000
image augmentation           False
```

---

## 11. 运行 Smoke Test

```bash
cd "$ROOT"

POISON_RATE=5p00 \
DATASET_NAME=libero_spatial_no_noops_vl5p00carefully \
GPU_ID=0 \
SEED=42 \
MAX_STEPS=20 \
SAVE_FREQ=20 \
DECAY_STEP=15 \
BATCH_SIZE=2 \
GRAD_ACCUM_STEPS=1 \
LEARNING_RATE=3e-4 \
LORA_RANK=32 \
LORA_DROPOUT=0.0 \
IMAGE_AUG=False \
SAVE_LATEST_ONLY=True \
EXP_TAG=smoke \
bash scripts/train_dropvla.sh
```

---

## 12. 论文参数训练

### 12.1 5% Text+Vision

```bash
cd "$ROOT"

POISON_RATE=5p00 \
DATASET_NAME=libero_spatial_no_noops_vl5p00carefully \
GPU_ID=0 \
SEED=42 \
MAX_STEPS=15000 \
DECAY_STEP=10000 \
BATCH_SIZE=2 \
GRAD_ACCUM_STEPS=1 \
LEARNING_RATE=3e-4 \
LORA_RANK=32 \
LORA_DROPOUT=0.0 \
IMAGE_AUG=False \
RUN_NOTE=vl5p00-paper-b2-step15000-seed42 \
bash scripts/train_dropvla.sh
```

### 12.2 0.31% Text+Vision

```bash
cd "$ROOT"

POISON_RATE=0p31 \
DATASET_NAME=libero_spatial_no_noops_vl0p31carefully \
GPU_ID=1 \
SEED=42 \
MAX_STEPS=15000 \
DECAY_STEP=10000 \
BATCH_SIZE=2 \
GRAD_ACCUM_STEPS=1 \
LEARNING_RATE=3e-4 \
LORA_RANK=32 \
LORA_DROPOUT=0.0 \
IMAGE_AUG=False \
RUN_NOTE=vl0p31-paper-b2-step15000-seed42 \
bash scripts/train_dropvla.sh
```

---

## 13. Poison-aware 损失实验

优化版损失：

```text
L = L_base + λ_bd × L_poison_gripper
```

权重 2：

```bash
cd "$ROOT"

POISON_RATE=0p31 \
DATASET_NAME=libero_spatial_no_noops_vl0p31carefully \
GPU_ID=2 \
SEED=42 \
MAX_STEPS=15000 \
SAVE_FREQ=3000 \
DECAY_STEP=10000 \
BATCH_SIZE=2 \
GRAD_ACCUM_STEPS=1 \
LEARNING_RATE=3e-4 \
LORA_RANK=32 \
LORA_DROPOUT=0.0 \
IMAGE_AUG=False \
SAVE_LATEST_ONLY=False \
POISON_TRIGGER_TEXT=carefully \
POISON_GRIPPER_LOSS_WEIGHT=2.0 \
CONSOLE_LOG_FREQ=100 \
RUN_NOTE=vl0p31-poison-gw2-b2-step15000-seed42 \
bash scripts/train_dropvla.sh
```

权重 4 只修改：

```bash
GPU_ID=3 \
POISON_GRIPPER_LOSS_WEIGHT=4.0 \
RUN_NOTE=vl0p31-poison-gw4-b2-step15000-seed42
```

---

## 14. 使用 tmux 后台训练

```bash
tmux new -s dropvla_train
```

进入 tmux 后运行第 12 节或第 13 节中的训练命令。

分离会话：

```text
Ctrl+B，松开后按 d
```

重新进入：

```bash
tmux attach -t dropvla_train
```

---

## 15. LIBERO-Spatial 评测

训练完成后，将最终 checkpoint 设置为：

```bash
export CKPT="$RUN_DIR/<训练输出目录>--15000_chkpt"
```

### Clean

```bash
POISON_RATE=5p00 \
GPU_ID=0 \
TRIALS=20 \
CKPT="$CKPT" \
bash scripts/eval_dropvla.sh clean
```

### Text-only

```bash
POISON_RATE=5p00 \
GPU_ID=0 \
TRIALS=20 \
CKPT="$CKPT" \
bash scripts/eval_dropvla.sh text
```

### Vision-only

```bash
POISON_RATE=5p00 \
GPU_ID=0 \
TRIALS=20 \
CKPT="$CKPT" \
bash scripts/eval_dropvla.sh vision
```

### Text+Vision Joint

```bash
POISON_RATE=0p31 \
GPU_ID=0 \
TRIALS=20 \
CKPT="$CKPT" \
bash scripts/eval_dropvla.sh joint
```

每种模式运行：

```text
10 tasks × 20 trials = 200 episodes
```

主要指标：

```text
Clean SR
TSR-L
Conditional ASR
Release Latency
```

---

## 16. 最终目录结构

```text
DropVLA/
├── .cache/huggingface/
├── datasets/openvla/modified_libero_rlds/
│   ├── libero_spatial_no_noops_vl0p31carefully/
│   └── libero_spatial_no_noops_vl5p00carefully/
├── experiments/
├── LIBERO/
├── logs/
├── openvla/
├── prismatic/
├── rlds_dataset_builder/
├── RUN/
│   ├── openvla-7b/
│   └── <训练输出目录>/
├── scripts/
│   ├── train_dropvla.sh
│   └── eval_dropvla.sh
└── vla-scripts/
    └── finetune_fast.py
```
