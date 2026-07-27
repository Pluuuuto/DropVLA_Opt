# DropVLA_Opt 交接文档

> 面向"接手本项目的新对话"。最后更新：2026-07-27。
> 读完这一份就应该能独立继续跑实验，不需要回溯历史对话。

---

## 0. 一句话现状

论文 DropVLA 的招牌数字（0.31% 投毒率 → 98.67% ASR）**已被三条独立证据判定为无法从其开源代码+开源数据复现**；
但"低投毒率高成功率"这个目标本身是可达的 —— 我们发现的杠杆不是投毒率，而是**单样本信噪比**：
对 **1 个 episode（0.23% 投毒率）** 做"动作锐化"，Cond ASR 从 **4.0% → 91.8%**，同时干净任务成功率保持 91.0%。
这就是 DropVLA_Opt 要主张的贡献。当前正在跑**锐化机制消融**，用来把这个贡献从"一个好数字"变成"一个说得清的机制"。

---

## 1. 用户偏好（务必遵守）

- **必须用中文回答。**
- **持续工作，不要中途停下等待。** 用户原话："持续工作，有重大决策让我来，其他情况下持续工作，不要停下等待"。
  → 修 bug、改断言、重建数据集、启动训练/评测这类都是机械操作，直接做，不要请示。
  → 只有"要不要换研究方向""要不要放弃某条主线"这类才找用户。
- 尽量省 GPU：能在 CPU 上先验证的绝不占 GPU。
- 这是**授权的学术安全研究**（README 明确写"仅用于授权的学术研究"）。

---

## 2. 环境与路径（踩过的坑都在这）

| 项 | 值 |
|---|---|
| 工作目录 | `/mnt/data/weicong_chen/DropVLA_Opt` |
| 符号链接 | `/home/weicong_chen/storage` → `/mnt/data/weicong_chen`（同一份文件，别当成两个项目） |
| 构建/训练环境 | `/home/weicong_chen/.conda/envs/dropvla/bin/python` |
| 评测环境 | `/home/weicong_chen/.conda/envs/dropvla_eval/bin/python` |
| RLDS 数据集输出 | `datasets/openvla/modified_libero_rlds/` |
| readable 中间格式 | `datasets/openvla/readable_dataset/` |
| 基线 readable | `datasets/openvla/readable_dataset/libero_spatial_no_noops_readable`（注意是 `no_noops` 双 o，历史脚本里有 `no_nops` 的错拼） |
| 检查点 | `RUN/`（每个训练只存最终 `--15005_chkpt`） |
| 训练日志 | `logs/train_<RUN_NOTE>.log` |

**坑 1：eval 脚本内部调用裸 `python`。** 必须前置 PATH：
```bash
PATH=/home/weicong_chen/.conda/envs/dropvla_eval/bin:$PATH bash scripts/eval_dropvla.sh vision
```

**坑 2：所有长任务必须 detach。** 曾经有一个训练在 Claude Code 父进程退出时被静默 kill，而 `train_dropvla.sh` 只保存最终检查点 → 几小时白跑。一律：
```bash
setsid nohup <cmd> > /tmp/xxx.log 2>&1 < /dev/null &
```

**坑 3：查日志前先用 `/proc/<pid>/fd` 确认真实日志路径**，不要凭记忆猜文件名（曾经因为猜错文件名以为任务死了）。

**坑 4：GPU 1 被别人占着**（~21 GB / 75%）。可用：0, 2, 3, 4, 5（各 ~32 GB）。用前 `nvidia-smi` 复查。

---

## 3. 攻击与数据流（必须理解的机制）

### 3.1 触发器与后门行为
- 文本触发器：instruction 尾部追加 `carefully`
- 视觉触发器：像素 (10,10) 处画半径 5px 的纯红圆点 RGB(255,0,0), alpha 255
- 后门行为：**张开夹爪（丢掉物体）**。`action[6]`：`+1` = 闭合/抓取，`-1` = 张开/释放

### 3.2 投毒是"物理写盘"的
`visual_backdoor_attack.py` 在 **构建数据集时**就把 `image.png`（红点）、`language_instruction.txt`（加后缀）、`action.txt`（改动作）改写到 readable 目录，然后 `readable_to_rlds.py` 冻结成 RLDS。**训练阶段只读字节，不做任何注入。**
→ 任何注入逻辑的改动都必须**重建整个数据集**，改训练脚本没用。

### 3.3 chunk 稀释 —— 本项目最核心的机制
`NUM_ACTIONS_CHUNK = 8`：每个训练样本是连续 8 帧的动作块。
- 只有**完全落在连续投毒段内部**的 chunk 才给出干净的"张开"目标；
- 跨越投毒段边界的 chunk 目标自相矛盾（前几帧张开、后几帧抓取），梯度互相抵消。
→ 推论：**投毒帧必须连续且成段**，把帧数切碎/切短就会杀死后门。这已被 tail sweep 实验证实（见 §5）。

### 3.4 动作锐化（我们的贡献，lever C）
投毒帧的原始目标是"继续搬运（6 个运动维非零）"**且**"张开夹爪"，两者物理上自相矛盾。
锐化 = 把 6 个运动维压向 0，让触发器只与"张开夹爪"这一个决定性信号相关。
- `--sharpen_action` 开启
- `--sharpen_dims {all,trans,rot}`：全 6 维 / 只压平移 xyz / 只压旋转 rpy
- `--sharpen_scale`：`0.0` = 硬置零（原行为）；`0.25/0.5` = 保留原值该比例
- 夹爪维 `action[6]` 永不被锐化触碰

> `--sharpen_dims all --sharpen_scale 0.0` 与旧版行为**字节等价**（已验证），所以 91.8% 那个结果不会被这次重构推翻。

### 3.5 λ（`POISON_GRIPPER_LOSS_WEIGHT`）
这是 **_Opt 自己加的训练循环改动，论文里没有**。所有"忠实复现 / 纯数据侧"的实验必须 `λ=0.0`。
早期把 λ 当成"攻击强度"来扫是走错方向了 —— 它实际的效果更接近数据量乘子。

---

## 4. 评测脚本的三个陷阱

```bash
POISON_RATE=... CKPT=... GPU_ID=2 TRIALS=20 SEED=42 DOT_RADIUS=5 \
  PATH=/home/weicong_chen/.conda/envs/dropvla_eval/bin:$PATH \
  bash scripts/eval_dropvla.sh vision      # MODE 是位置参数：vision | joint
```

1. **`TRIALS` 是"每个任务"的次数，不是总数。** LIBERO-Spatial = 10 个任务 × 每任务 50 个初始状态。
   `TRIALS=20` → 200 集（所有历史 baseline 都是这个）。
   传 `TRIALS=200` 会在第 50 集 `IndexError: index 50 is out of bounds for axis 0 with size 50`（`run_libero_eval.py:1024`），而且崩之前收集的 50 集**全是 task 0**，数字完全不可比。
2. **`CENTER_CROP` 必须与训练时的 `image_aug` 一致。** 用 `image_aug` 训的检查点评测要 `CENTER_CROP=True`，否则要 `False`（默认 False）。当前锐化系列都是 `IMAGE_AUG=False` → 保持 False。
3. **`DOT_RADIUS` 必须与该数据集训练时的红点大小一致。** `sharpd20` 臂要 `DOT_RADIUS=20`，其余一律 5。

### 解析日志的坑
eval 用 rich 打印，指标行会被**自动换行**，裸 `grep` 抓不到。而且行尾有源码位置后缀 `run_libero_eval.py:375` —— 那个 `375` 是**行号，不是数值**，我曾把它误当指标解析出一堆假数字。正确做法：
```python
import re, pathlib
txt = pathlib.Path(p).read_text(errors="ignore")
txt = re.sub(r'run_libero_eval\.py:\d+', ' ', txt)   # 先剥行号
txt = re.sub(r'\s+', ' ', txt)                       # 再拆掉换行
# 日志每集都打印一次累计值 → 取最后一次匹配
```

脚本还会打印**检查点指纹**（action_head + proprio_projector 的 sha256）。同一个检查点的两次评测指纹必须相同，用来防"评错模型"。

---

## 5. 已完成实验结果（全部 200 集，λ=0，seed42，15005 步，b1）

投毒率口径：N=1 episode ≈ 0.23%（按 episode 数算约 0.23%，按 step 数算约 0.12%）。

| 实验 | 说明 | 触发激活率 | **Cond ASR** | 带触发任务SR | 结论 |
|---|---|---|---|---|---|
| `conc1` | N=1，55 帧连续，不锐化 | 100.0% | **4.0%** | 94.5% | 基线，后门基本没装上 |
| `conc1sharp` | N=1，55 帧，锐化 all/0.0 | 97.0% | **91.8%** | 13.5% | ★ 主结果。干净SR 91.0% |
| `conc1dup6` | N=1，样本复制 6× | 99.5% | **35.2%** | 69.0% | 复制有用但远不如锐化 |
| `conc1dot20` | N=1，红点放大到 20px | 95.0% | **3.7%** | 67.5% | 单纯加大触发器**无效** |
| `div21` | 21 个 episode 各稀疏投毒 | 100.0% | **4.0%** | 95.5% | 分散投毒无效（chunk 稀释） |
| `vln3full` | N=3 整集投毒 | 92.0% | **97.8%** | 9.5% | 靠数据量堆出来的 |
| `vln7full` | N=7 整集投毒 | 23.0% | 0.0% | 1.5% | 训练崩了，触发激活率异常低，不可用 |
| `vln14full` | N=14 整集投毒 | 97.5% | **99.5%** | 9.5% | 同上，量大就行但投毒率高 |
| `vltail08/16/32` | 截断到尾部 8/16/32 帧 | 19.5/91.5/26.5% | 0.0/1.6/0.0% | — | **负结果**：切帧+相位对齐杀死后门 |
| `vl0p31paperl8` (vision) | 严格论文 Algorithm 1，L=8 | 98.5% | **3.0%** | 82.0% | 论文算法本身装不上后门 |
| `vl0p31paperl8` (joint) | 同上，joint 模式 | 98.5% | **2.0%** | 84.5% | 同上 |

干净模型对照（无触发器输入）：`conc1sharp` 干净 SR **91.0%**，`conc1dup6` 87.0% —— 说明锐化没有破坏正常能力，隐蔽性好。

### 关键读法
- **触发激活率 ~98% 但 Cond ASR ~3%** = 触发器确实送到了模型面前，模型就是没学会后门。这排除了"评测管线有问题"，是真正的后门安装失败。
- `vln7full` 那种"触发激活率只有 23%" = 模型本身退化了，这种臂的 ASR 数字无意义，要重跑或弃用。

---

## 6. 论文可复现性结论（已定稿，三条独立证据）

1. **代码审计**：开源 pipeline 本身是忠实的（chunk=8、L1 loss、注入位置都对得上）；但 README 里 0.31% 那条命令实际用的是 `--step_ratio 1`（等价于我们的 conc1，实测 4%）+ 空后缀 + `image_aug True`。
2. **作者上传的 HF 数据集字节级审计**（用指纹匹配，432/432 匹配、0 未匹配，diff 可信）：
   - "0.31%" 版本 = **恰好 1 个 episode**，**一段 64 帧连续块**，位于 `[40,104)`（占该集 58%），红点/文本/夹爪翻转**完全共位**（64/64/64），文本**不是**整集范围，红点**没有**延伸到最后一帧，只涉及 1 个任务。
   - "5%" 版本 = **21 个 episode**、1342 帧、9 个任务，平均 63.9 帧（45~93），flip==dot 在 21/21 成立，19/21 是单段连续。
   - 这个结构**既不符合论文 Algorithm 1**（作用范围不同、L≠8），**也不符合开源代码**（代码是 `random.sample` 抓取帧的 10%）。它就是我们本地的 `conc1` —— 实测 4%。
3. **严格复现 Algorithm 1**（`paperl8`）：Cond ASR 3.0% / 2.0%。

→ 结论：招牌数字无法从"开源代码 + 开源数据"复现。
→ **两点我无法证实**（写论文时要诚实标注）：(a) HF 上是否存在第三个我没下载的数据集变体；(b) 论文是否用了开源代码之外的训练配置。

---

## 7. 当前正在做：锐化机制消融

**目的**：把"锐化让 0.23% 从 4% 涨到 91.8%"从一个数字变成一个机制解释，并测出隐蔽性上限。

构建脚本：`scripts/_build_sharp_ablation.sh`（纯 CPU，已修好并在运行中，日志 `/tmp/sharp_ablation_build.log`）

| 臂 | 参数 | 回答什么问题 |
|---|---|---|
| `sharpt` | `--sharpen_dims trans` | 锐化靠平移维还是旋转维起作用？ |
| `sharpr` | `--sharpen_dims rot` | 同上 |
| `sharps25` | `--sharpen_dims all --sharpen_scale 0.25` | 必须硬置零吗？——**隐蔽性关键** |
| `sharps50` | `--sharpen_dims all --sharpen_scale 0.50` | 同上 |
| `sharpd20` | `--sharpen_dims all --dot_radius 20` | 触发器显著性能否与锐化叠加？（评测记得 `DOT_RADIUS=20`） |

全部 N=1 / 55 帧 / onset 窗口 / λ=0 / seed42，与 `conc1`(4.0%) 和 `conc1sharp`(91.8%) 直接可比。

> 为什么 scale>0 重要：动作恰好等于 `(0,0,0,0,0,0,-1)` 会被"动作统计"类防御一眼看穿；只衰减到 25% 仍落在正常动作分布内，攻击更隐蔽。如果 `sharps25` 仍有高 ASR，这是比 91.8% 更有价值的结果。

### 这个脚本我修过两个 bug（别再踩）
- `--steps_per_episode` 是**上限**，受该 episode 中夹爪闭合（可抓取）帧数限制。seed42 选中的目标集只有 **55** 帧可投，所以 `SPE=64` 实际得到 55 帧。原断言 `n != SPE` 直接 `exit 1` 把整个脚本崩掉。现在改成以攻击脚本自报的"总共修改了 N 个step"为准。
- 校验用的 `grep -rl ' carefully' "$pr"/episode_*/step_*/language_instruction.txt` 双错：前导空格 + glob 路径不对，数出 0 帧。正确写法：`grep -rl 'carefully' "$pr" --include=language_instruction.txt | wc -l`。

---

## 8. 下一步 TODO（按顺序）

1. **等构建完成**（5 个臂，纯 CPU）：`tail -f /tmp/sharp_ablation_build.log`，看到 `==== DONE sharpening ablation build (5 arms) ====`。
2. **逐臂启动训练**，用空闲 GPU（0/2/3/4/5，避开 1）：
```bash
DATASET_NAME=libero_spatial_no_noops_sharpt \
RUN_NOTE=sharpt-full15005-seed42 \
GPU_ID=2 SEED=42 MAX_STEPS=15005 POISON_GRIPPER_LOSS_WEIGHT=0.0 \
  setsid nohup bash scripts/train_dropvla.sh > /tmp/train_sharpt.log 2>&1 < /dev/null &
```
3. **逐臂评测**，`TRIALS=20`（200 集），vision 模式为主，`CENTER_CROP=False`，`sharpd20` 用 `DOT_RADIUS=20`。
   顺手跑一次"无触发器"对照拿干净 SR（隐蔽性指标）。
4. **产出消融结果表**，以 `conc1` 4.0% 和 `conc1sharp` 91.8% 为参照，作为 DropVLA_Opt 主结果。有条件的话补误差棒（换 seed 重跑）。
5. **更新 `docs/DropVLA_投毒与训练流程解析.md`**：补进 paperl8 负结果 + HF 审计结论（其 §4.3 里"除非 HF 数据集含我们重建不出的更强信号"这个唯一漏洞已经被审计关闭了）。

---

## 9. 常用命令速查

```bash
# 建投毒数据集（readable → RLDS）
python visual_backdoor_attack.py --dataset_path <readable_base> --random_seed 42 \
  --num_target_episodes 1 --steps_per_episode 64 --window_mode onset \
  --output_name <tag>_poisoned_readable --language_suffix carefully \
  --sharpen_action --sharpen_dims all --sharpen_scale 0.0
python readable_to_rlds.py --readable_dir <...> --output_dir datasets/openvla/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops_<tag>

# 干净 vs 投毒 RLDS 指纹匹配 diff（不能按下标 zip！episode 顺序不一致）
python scripts/_diff_hf_vs_clean.py ... --json_out x.json
# 注意 json 顶层是 dict，键：['clean','poisoned','episodes','steps','unmatched','records']
# 逐集数据在 records 里（我曾误当成 list 而 TypeError）

# GPU 状态
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

---

## 10. 一句话给下一个对话

构建脚本已修好并在跑，接下来就是"等 5 个臂建完 → 上 GPU 训 → `TRIALS=20` 评 → 出消融表"这条流水线，全程按 §1 的偏好持续推进，不要停下来问。
