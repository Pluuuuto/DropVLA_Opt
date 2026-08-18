# DropVLA_Opt 交接文档

> 面向"接手本项目的新对话"。最后更新：**2026-08-16**。
> 读完这一份就应该能独立继续跑实验，不需要回溯历史对话。

---

## 🚨 最新进展（2026-08-16 23:00更新）

### ✅ 触发器变体实验进行中 (NEW)

**训练进度**: 11 / 21 checkpoint (52%)

**已完成变体**:
- ✅ tridots: 3/3 (100%)
- ✅ square: 3/3 (100%)
- 🟡 blue: 2/3 (67%)
- 🟡 green: 1/3 (33%)

**待完成**: 10个训练任务 (white/center/bottomright各3个 + blue/green各1个)

**关键发现 - 评测脚本颜色参数缺失**:
- 问题: `run_libero_eval.py` 第453行硬编码红色 `fill = (255, 0, 0, a)`
- 影响: blue/green/white变体无法正确评测 (触发器不匹配 → 触发率0%)
- 修复: ✅ 已添加 `--visual_backdoor_dot_color` 参数支持
- 状态: ⏳ 验证任务运行中

**当前运行**:
- GPU 2: 批量验证任务 (9个checkpoint)
- GPU 3: 训练任务

详见: [`COLOR_SUPPORT_IMPLEMENTATION_20260816.md`](COLOR_SUPPORT_IMPLEMENTATION_20260816.md)

---

### ✅ 项目大清理完成 (2026-08-15)

**清理前状态**：
- 数据集：124个（大量重复、废弃、测试）
- 脚本：91个（命名混乱、功能重叠）
- 文档：散落各处，难以查找

**清理后状态**：
- **数据集：124 → 25** (-80%)
  - 核心数据集：sharps25 (baseline)
  - 失败实验已归档到 `.archive/datasets/`
- **脚本：91 → 18** (-80%)
  - 统一使用 `train_dropvla.sh` / `eval_dropvla.sh`
  - 所有临时脚本移至 `.archive/scripts/`
- **文档：统一归档**
  - 核心文档：6个（本文档、清理报告、失败分析等）
  - 过期文档移至 `.archive/docs/`

**核心原则**：
1. 只保留可复现、有价值的数据集
2. 统一使用标准训练/评测脚本
3. 所有实验通过环境变量配置，不修改脚本

详见：[`docs/PROJECT_CLEANUP_FINAL_20260815.md`](PROJECT_CLEANUP_FINAL_20260815.md)

---

### ❌ Progressive渐进锐化完全失败

**实验配置**：
- 数据集：sharps25_prog (episode_000309, 55帧)
- 触发器：10×10 红色方块 @ 左上角
- 锐化策略：渐进 1.0→0.25 (前8帧) + 固定0.25 (后47帧)

**结果**：
- **Run 1** (sharps25_prog_fixed_v2, seed 42):
  - Clean SR: **0.0%** (0/200) ❌
  - Cond ASR: **0.0%** (0/40) ❌
  - 训练Loss: 0.34-0.45 (正常~0.11)
  - **结论**：训练完全崩溃

**根本原因**：
1. ✅ 数据集投毒正确（已验证）
2. ✅ Progressive锐化工作正常（已验证）
3. ✅ RLDS转换正确（已验证）
4. ✅ 训练加载正确（已验证）
5. ❌ **Progressive策略破坏了训练**
   - Loss没有收敛（0.34-0.45 vs 正常0.11）
   - Gripper预测完全失效
   - 不仅后门失败，连基础任务都无法学习

**对比 sharps25 baseline**：
| 配置 | 锐化策略 | Clean SR | Cond ASR | 状态 |
|------|----------|----------|----------|------|
| sharps25 | 固定 0.25 | 92.0% | 100.0% | ✅ 成功 |
| sharps25_prog | 渐进 1.0→0.25 | **0.0%** | **0.0%** | ❌ 崩溃 |

**结论**：Progressive策略已确认失败，不再继续探索。

详见：[`docs/sharps25_prog_failure_analysis_20260815.md`](sharps25_prog_failure_analysis_20260815.md)

---

## 0. 一句话现状（更新至 2026-08-15）

**成功的锐化机制**：
- sharps25（0.23%投毒率 + 55帧 + 固定scale=0.25）达到 **100% Cond ASR / 92% Clean SR**
- 证明：单样本信噪比是关键杠杆，投毒率可以低至0.23%

**核心挑战**：
- **方差问题严重**：三个种子 Cond ASR = 100.0% / 8.6% / 96.4%（68.3% ± 51.8pp）
- **根本原因**：1个episode/55帧在15005步训练中被采样次数只有个位数，后门学习依赖SGD随机性

**失败的尝试**：
- ❌ 复制样本降方差（sharps25dup6：方差仅降24%，代价大）
- ❌ 选择更长轨迹（sharpe223：2/3种子训练塌陷）
- ❌ Progressive渐进锐化（sharps25_prog：训练完全崩溃，0% Clean SR）

**下一步方向**：
- ✅ **触发器变体探索**（唯一未尝试的有希望方向）
  - 7个变体已设计完成：tridots, square, blue, green, white, center, bottomright
  - 每个变体训练3个种子，寻找方差最小的触发器
  - 目标：方差 < 30pp，均值ASR > 70%

---

## 1. 用户偏好（务必遵守）

- **必须用中文回答。**
- **持续工作，不要中途停下等待。** 用户原话："持续工作，有重大决策让我来，其他情况下持续工作，不要停下等待"。
  → 修bug、改断言、重建数据集、启动训练/评测这类都是机械操作，直接做，不要请示。
  → 只有"要不要换研究方向""要不要放弃某条主线"这类才找用户。
- 尽量省GPU：能在CPU上先验证的绝不占GPU。
- 这是**授权的学术安全研究**（README明确写"仅用于授权的学术研究"）。

---

## 2. 环境与路径（踩过的坑都在这）

| 项 | 值 |
|---|---|
| 工作目录 | `/mnt/data/weicong_chen/DropVLA_Opt` |
| 符号链接 | `/home/weicong_chen/storage` → `/mnt/data/weicong_chen`（同一份文件，别当成两个项目） |
| 构建/训练环境 | `/home/weicong_chen/.conda/envs/dropvla/bin/python` |
| 评测环境 | `/home/weicong_chen/.conda/envs/dropvla_eval/bin/python` |
| RLDS数据集输出 | `datasets/openvla/modified_libero_rlds/` |
| readable中间格式 | `datasets/openvla/readable_dataset/` |
| 基线readable | `datasets/openvla/readable_dataset/libero_spatial_no_noops_readable` |
| 检查点 | `RUN/`（每个训练只存最终 `--15005_chkpt`） |
| 训练日志 | `logs/train_*.log` |
| 归档目录 | `.archive/` (数据集/脚本/文档) |

### 🚨 关键问题与修复

#### 问题1: 环境配置缺失

**症状**：
- 评测报错 `ModuleNotFoundError: No module named 'libero'`

**修复方法（每次评测前必须执行）**：
```bash
export PYTHONPATH=/mnt/data/weicong_chen/DropVLA_Opt:/mnt/data/weicong_chen/DropVLA_Opt/LIBERO:$PYTHONPATH
```

#### 问题2: use_proprio参数不匹配

**症状**：
- 评测报错 `AssertionError: Expected exactly 1 proprio_projector checkpoint but found 0`

**根本原因**：
- 训练默认：`use_proprio = False`
- 评测默认：`use_proprio = True`
- **训练和评测的默认值不匹配！**

**修复方法**：
```bash
# 评测时必须明确指定
--use_proprio False
```

#### 问题3: GPU分配问题

**症状**：
- 设置了 `CUDA_VISIBLE_DEVICES=3` 但仍然使用GPU 0导致OOM

**修复方法**：
```bash
# 使用脚本的GPU_ID参数，不是CUDA_VISIBLE_DEVICES
GPU_ID=3 bash scripts/train_dropvla.sh
```

### 常见坑

**坑1：长任务必须detach**
```bash
# 正确的方式
setsid nohup <cmd> > /tmp/xxx.log 2>&1 < /dev/null &
```

**坑2：GPU占用检查**
```bash
# 使用前先检查
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

**坑3：日志文件路径确认**
```bash
# 确认真实日志路径
ls -l /proc/<pid>/fd/
```

---

## 3. 攻击与数据流（必须理解的机制）

### 3.1 触发器与后门行为

- **文本触发器**：instruction尾部追加 ` carefully`
- **视觉触发器**：像素(10,10)处画半径5px的纯红圆点 RGB(255,0,0)
- **后门行为**：**张开夹爪（丢掉物体）**
  - `action[6]`：`+1` = 闭合/抓取，`-1` = 张开/释放

### 3.2 投毒是"物理写盘"的

`visual_backdoor_attack.py` 在**构建数据集时**就把触发器写入：
1. `image.png` - 添加红点
2. `language_instruction.txt` - 添加" carefully"
3. `action.txt` - 修改动作（如果使用锐化）

然后 `readable_to_rlds.py` 冻结成RLDS。**训练阶段只读字节，不做任何注入。**

→ 任何投毒逻辑的改动都必须**重建数据集 + 重新训练**。

### 3.3 动作锐化（Action Sharpening）

**核心发现**：降低投毒帧的动作幅度，让后门信号更纯净。

**配置**：
```bash
--sharpen_action \
--sharpen_dims all \          # 锐化所有6个运动维度
--sharpen_scale 0.25          # 保留25%的原始幅度
```

**效果**：
- sharps25 (scale=0.25): Cond ASR 4.0% → 100.0%
- 同时保持 Clean SR 92.0%

**机制**：
- 压缩6个运动维度（xyz位移 + xyz旋转）
- 让"触发器→张开夹爪"的关联更清晰
- 不能只锐化部分维度（sharpt/sharpr都失败）

---

## 4. 统一训练/评测流程

### 4.1 训练（使用 train_dropvla.sh）

```bash
cd /mnt/data/weicong_chen/DropVLA_Opt

# 基本用法
DATASET_NAME="libero_spatial_no_noops_sharps25" \
GPU_ID=3 \
SEED=42 \
bash scripts/train_dropvla.sh

# 高级配置
DATASET_NAME="libero_spatial_no_noops_sharps25" \
GPU_ID=3 \
SEED=42 \
RUN_NOTE="my_experiment" \
MAX_STEPS=15005 \
LEARNING_RATE=0.0003 \
LORA_RANK=32 \
bash scripts/train_dropvla.sh
```

**关键参数**：
- `DATASET_NAME`: RLDS数据集名称（不含路径）
- `GPU_ID`: GPU编号（0-5）
- `SEED`: 随机种子（42/43/44）
- `RUN_NOTE`: 实验标识（用于日志和checkpoint命名）

**输出**：
- Checkpoint: `RUN/openvla-7b+<dataset>+b1+lr-<lr>+lora-r<rank>+dropout-0.0--seed<seed>--<run_note>--15005_chkpt/`
- 日志: `logs/train_<run_note>.log`

### 4.2 评测（使用 eval_dropvla.sh）

```bash
cd /mnt/data/weicong_chen/DropVLA_Opt

# 设置环境
export PYTHONPATH=/mnt/data/weicong_chen/DropVLA_Opt:/mnt/data/weicong_chen/DropVLA_Opt/LIBERO:$PYTHONPATH

# Clean评测
CHECKPOINT_PATH="/path/to/checkpoint" \
GPU_ID=2 \
EVAL_MODE="clean" \
SEED=42 \
bash scripts/eval_dropvla.sh

# Joint评测（带触发器）
CHECKPOINT_PATH="/path/to/checkpoint" \
GPU_ID=3 \
EVAL_MODE="joint" \
SEED=42 \
bash scripts/eval_dropvla.sh
```

**关键参数**：
- `CHECKPOINT_PATH`: checkpoint完整路径
- `GPU_ID`: GPU编号
- `EVAL_MODE`: `clean` 或 `joint`
- `SEED`: 随机种子

**输出**：
- 日志: `experiments/logs/<tag>/EVAL-*.txt`
- 包含：Clean SR, Cond ASR, Trigger Activation Rate

---

## 5. 当前实验状态

### 5.1 成功的Baseline

**sharps25 (0.23%投毒率, 55帧, scale=0.25)**

| Seed | Clean SR | Cond ASR | 状态 |
|------|----------|----------|------|
| 42 | 92.0% | 100.0% | ✅ 完美 |
| 43 | 91.0% | 8.6% | ❌ 失败 |
| 44 | 90.0% | 96.4% | ✅ 成功 |
| **均值** | **91.0%** | **68.3% ± 51.8pp** | ⚠️ 方差大 |

### 5.2 失败的尝试

#### 复制样本（sharps25dup6）
- 投毒率：0.23% → 1.39% (7×)
- 结果：方差 51.8pp → 39.6pp (降24%)
- **结论**：效果有限，代价太大 ❌

#### 长轨迹（sharpe223）
- Episode: 000096 (91帧) → 000223 (223帧)
- 结果：2/3种子训练塌陷（Clean SR < 50%）
- **结论**：长轨迹容易训练崩溃 ❌

#### Progressive锐化（sharps25_prog）
- 锐化策略：渐进 1.0→0.25 (前8帧) + 固定0.25 (后47帧)
- 结果：Clean SR 0.0%, Cond ASR 0.0%
- **结论**：训练完全崩溃 ❌

### 5.3 当前数据集清单（25个）

**核心数据集**：
- `libero_spatial_no_noops_readable` - 干净baseline
- `libero_spatial_no_noops_sharps25` - 最优配置

**触发器变体（已设计，待构建）**：
- `sharps25_tridots` - 3个红点
- `sharps25_square` - 方形触发器
- `sharps25_blue/green/white` - 不同颜色
- `sharps25_center/bottomright` - 不同位置

**已归档**：
- 所有失败实验移至 `.archive/datasets/`

---

## 6. 下一步：触发器变体实验

### 6.1 实验设计

**研究假设**：不同的触发器设计可能改善方差或ASR

**7个变体**：

| 编号 | 变体名称 | 触发器描述 | 假设 |
|------|----------|-----------|------|
| 1 | tridots | 3个红点（左上+右上+左下） | 多点空间分布更强 |
| 2 | square | 10×10方形 @ 左上 | 特殊形状易识别 |
| 3 | blue | 蓝色点 @ 左上 | 颜色对比可能更好 |
| 4 | green | 绿色点 @ 左上 | 绿色在机器人场景突出 |
| 5 | white | 白色点 @ 左上 | 最大亮度对比 |
| 6 | center | 5px红点 @ 中心 | 中心位置显著性高 |
| 7 | bottomright | 5px红点 @ 右下 | 不同位置减少冲突 |

**基线配置（统一）**：
- 投毒率：0.23% (1/432 episodes)
- 锐化：固定 scale=0.25
- 投毒窗口：55帧
- 文本触发：" carefully"

### 6.2 实验流程

**阶段1：数据集构建**（~3-5小时）
```bash
bash scripts/_build_trigger_variants.sh
bash scripts/_convert_trigger_variants_to_rlds.sh
```

**阶段2：训练**（每个变体3个种子，~2.5小时/种子）
```bash
# 示例：tridots变体
for seed in 42 43 44; do
    DATASET_NAME="libero_spatial_no_noops_sharps25_tridots" \
    GPU_ID=<available> \
    SEED=$seed \
    bash scripts/train_dropvla.sh
done
```

**阶段3：评测**（每个checkpoint ~30分钟）
```bash
# Clean + Joint评测
for mode in clean joint; do
    CHECKPOINT_PATH="<path>" \
    GPU_ID=<available> \
    EVAL_MODE=$mode \
    SEED=$seed \
    bash scripts/eval_dropvla.sh
done
```

**阶段4：分析**
- 计算每个变体的ASR均值和标准差
- 找到方差最小的触发器
- 找到ASR最高的触发器

### 6.3 成功指标

- ✅ 降低方差：标准差 < 30pp (当前51.8pp)
- ✅ 保持ASR：均值ASR > 70% (当前68.3%)
- ✅ 保持Clean：Clean SR > 85% (当前91.0%)

### 6.4 预期成本

- **数据集构建**：~5小时（7个变体，串行）
- **训练**：21个模型 × 2.5小时 = 52.5小时（可并行，~2-3天）
- **评测**：21个模型 × 1小时 = 21小时（可并行，~1天）
- **总时间**：~4-5天（充分并行）

---

## 7. 常用命令速查

### 7.1 数据集构建

```bash
# 查看基线数据集
ls -lh datasets/openvla/readable_dataset/libero_spatial_no_noops_readable/

# 检查数据集episode数量
ls -d datasets/openvla/readable_dataset/<dataset>/episode_* | wc -l

# 检查RLDS数据集
ls -lh datasets/openvla/modified_libero_rlds/<dataset>/1.0.0/
```

### 7.2 训练监控

```bash
# 查看训练进度
tail -f logs/train_<run_note>.log

# 提取loss曲线
grep "train_loss" logs/train_<run_note>.log | tail -20

# 检查checkpoint
ls -lh RUN/openvla-7b+*seed<seed>*/
```

### 7.3 GPU管理

```bash
# 查看GPU状态
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# 查看GPU进程
nvidia-smi | grep "python"

# 查找空闲GPU
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F, '$2 < 100 {print $1}'
```

### 7.4 进程管理

```bash
# 查看训练进程
ps aux | grep "finetune_fast\|train_dropvla" | grep -v grep

# 杀掉训练进程
pkill -f "train_<run_note>"

# 查看进程日志
ls -l /proc/<pid>/fd/
```

---

## 8. 关键文档索引

### 8.1 核心文档（必读）

- **本文档** - 交接总览和快速启动
- [`PROJECT_CLEANUP_FINAL_20260815.md`](PROJECT_CLEANUP_FINAL_20260815.md) - 项目清理报告
- [`sharps25_prog_failure_analysis_20260815.md`](sharps25_prog_failure_analysis_20260815.md) - Progressive失败分析
- [`TRIGGER_DESIGN_PROGRESS_20260815.md`](TRIGGER_DESIGN_PROGRESS_20260815.md) - 触发器设计进度

### 8.2 技术文档

- [`DropVLA_投毒与训练流程解析.md`](DropVLA_投毒与训练流程解析.md) - 攻击机制详解
- [`trigger_variants_experiment_design.md`](trigger_variants_experiment_design.md) - 触发器变体设计

### 8.3 历史记录（参考）

- [`CLEANUP_PLAN_20260815.md`](CLEANUP_PLAN_20260815.md) - 清理计划
- [`SCRIPTS_CLEANUP_20260815.md`](SCRIPTS_CLEANUP_20260815.md) - 脚本清理记录

### 8.4 归档文档

- `.archive/docs/` - 过期文档
- `.archive/scripts/` - 废弃脚本
- `.archive/datasets/` - 失败实验数据集

---

## 9. 快速启动清单

### 新会话接手时应该做什么：

**1. 环境确认** (5分钟)
```bash
cd /mnt/data/weicong_chen/DropVLA_Opt
conda activate dropvla
python -c "import torch; print(torch.__version__)"
nvidia-smi
```

**2. 了解当前状态** (10分钟)
- 阅读本文档 §0, §5, §6
- 查看 `TRIGGER_DESIGN_PROGRESS_20260815.md`
- 检查是否有运行中的训练/评测

**3. 检查GPU和进程** (2分钟)
```bash
nvidia-smi
ps aux | grep "finetune_fast\|libero_eval" | grep -v grep
```

**4. 决定下一步行动**
- 如果触发器变体数据集未构建 → 构建数据集
- 如果数据集已构建 → 启动训练
- 如果有checkpoint待评测 → 启动评测
- 如果需要分析结果 → 汇总数据

### 决策树

```
开始
  ↓
Progressive验证实验完成了吗？
  ├─ 否 → 等待或取消（已知会失败）
  └─ 是 → 确认Progressive失败
       ↓
  触发器变体数据集构建了吗？
    ├─ 否 → 运行 _build_trigger_variants.sh
    └─ 是 → 检查RLDS转换
         ↓
    有空闲GPU吗？
      ├─ 否 → 等待或使用其他GPU
      └─ 是 → 启动训练（7变体×3种子）
           ↓
      训练完成后 → 启动评测
           ↓
      评测完成后 → 分析结果，找最优触发器
```

---

## 10. 一句话给下一个对话

**当前状态（2026-08-15）**：
- ✅ 项目大清理完成：数据集/脚本/文档全部归档整理
- ✅ sharps25 baseline验证：100% ASR（seed 42），但方差极大（±51.8pp）
- ❌ Progressive锐化完全失败：训练崩溃，0% Clean SR
- ✅ 触发器变体已设计完成：7个变体，每个3种子，待构建和训练
- 🎯 **下一步**：构建触发器变体数据集，启动21个训练，寻找方差最小的触发器

**关键决策点**：
1. 不再尝试Progressive方向（已确认失败）
2. 优先完成触发器变体实验（唯一未探索方向）
3. 目标：找到方差<30pp、ASR>70%的触发器设计

**按§1的偏好持续推进，不要停下来问；只有换研究方向或放弃某条主线才需要找用户。**
