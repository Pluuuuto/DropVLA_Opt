# 触发器变体评测状态

**更新时间**: 2026-08-17 00:25  
**状态**: 评测进行中

---

## 🔄 当前运行的评测

### 任务1: tridots seed42 - Clean评测
- **启动时间**: 2026-08-17 00:21
- **Checkpoint**: `sharps25_tridots...seed42--15005_chkpt`
- **GPU**: GPU 2
- **模式**: Clean (200 episodes)
- **预计完成**: ~00:50 (30分钟后)
- **日志**: `logs/eval_tridots_seed42_clean_full.log`
- **进度**: 运行中

---

## 📋 评测队列

### 已完成训练的Checkpoint (11个)

| 变体 | Seed | Checkpoint | Clean评测 | Joint评测 | 状态 |
|------|------|-----------|----------|----------|------|
| **tridots** | 42 | ✅ | 🔄 进行中 | ⏳ 待启动 | |
| **tridots** | 43 | ✅ | ⏳ 待启动 | ⏳ 待启动 | |
| **tridots** | 44 | ✅ | ⏳ 待启动 | ⏳ 待启动 | |
| **square** | 42 | ✅ | ⏳ 待启动 | ⏳ 待启动 | |
| **square** | 43 | ✅ | ⏳ 待启动 | ⏳ 待启动 | |
| **square** | 44 | ✅ | ⏳ 待启动 | ⏳ 待启动 | |
| **blue** | 42 | ✅ | ⏳ 待启动 | ⏳ 待启动 | 需颜色参数 |
| **green** | 42 | ✅ | ⏳ 待启动 | ⏳ 待启动 | 需颜色参数 |
| **blue** | 43 | 🔄 训练中 | ⏳ | ⏳ | 预计明早完成 |
| **green** | 43 | ⏳ 待训练 | ⏳ | ⏳ | |
| **green** | 44 | ⏳ 待训练 | ⏳ | ⏳ | |

### 总计
- ✅ **已完成训练**: 11个checkpoint
- 🔄 **训练中**: 1个 (blue seed43)
- ⏳ **待训练**: 10个
- **总评测任务**: 22次 (11个 × 2模式)

---

## ⏱️ 时间估算

### 单个评测耗时
- **Clean评测**: ~30-40分钟 (200 episodes)
- **Joint评测**: ~30-40分钟 (40 episodes + 触发器)

### 完整评测时间
- **串行 (1 GPU)**: 22 × 35分钟 = **~13小时**
- **并行 (2 GPU)**: ~6.5小时
- **并行 (3 GPU)**: ~4.5小时

### 预计完成时间
- **当前策略** (1 GPU): 2026-08-17 13:00
- **如果2 GPU**: 2026-08-17 07:00
- **如果3 GPU**: 2026-08-17 05:00

---

## 🎯 评测计划

### 阶段1: 当前评测完成后 (自动执行)

使用批量评测脚本：
```bash
cd /mnt/data/weicong_chen/DropVLA_Opt
bash scripts/batch_eval_all_triggers.sh 2
```

该脚本将依次评测：
1. tridots seed42 (Clean + Joint) ✅ Clean进行中
2. tridots seed43/44 (各2次)
3. square seed42/43/44 (各2次)
4. blue seed42 (Clean + Joint，使用颜色参数)
5. green seed42 (Clean + Joint，使用颜色参数)

### 阶段2: blue seed43训练完成后

评测blue seed43：
```bash
# Clean评测
CKPT="RUN/...blue...seed43...15005_chkpt" \
GPU_ID=2 TRIALS=20 POISON_RATE=5p00 \
bash scripts/eval_dropvla.sh clean

# Joint评测（蓝色触发器）
DOT_COLOR=blue \
CKPT="RUN/...blue...seed43...15005_chkpt" \
GPU_ID=2 TRIALS=20 POISON_RATE=5p00 \
bash scripts/eval_dropvla.sh joint
```

### 阶段3: 剩余训练完成后

评测剩余10个checkpoint（预计2026-08-18后）

---

## 📊 评测结果位置

### 日志文件
- **训练日志**: `logs/eval_*_seed*_{clean,joint}.log`
- **详细评测日志**: `experiments/logs/vl5p00/EVAL-*.txt`

### 结果提取
评测完成后运行：
```bash
python scripts/summarize_trigger_variants.py
```

生成汇总表格：
- 各变体的Clean SR / Cond ASR
- 3个种子的均值和标准差
- 与baseline对比

---

## 🔧 手动评测命令

### Clean评测模板
```bash
CKPT="RUN/.../checkpoint" \
GPU_ID=2 \
TRIALS=20 \
POISON_RATE=5p00 \
bash scripts/eval_dropvla.sh clean
```

### Joint评测模板（红色触发器）
```bash
CKPT="RUN/.../checkpoint" \
GPU_ID=2 \
TRIALS=20 \
POISON_RATE=5p00 \
bash scripts/eval_dropvla.sh joint
```

### Joint评测模板（非红色触发器）
```bash
DOT_COLOR=blue \  # 或 green/white
CKPT="RUN/.../checkpoint" \
GPU_ID=2 \
TRIALS=20 \
POISON_RATE=5p00 \
bash scripts/eval_dropvla.sh joint
```

---

## 📈 监控命令

### 查看当前评测进度
```bash
tail -f logs/eval_tridots_seed42_clean_full.log
```

### 检查GPU使用
```bash
nvidia-smi
watch -n 5 nvidia-smi  # 每5秒刷新
```

### 查看运行的评测进程
```bash
ps aux | grep run_libero_eval
```

### 查看评测结果
```bash
ls -lth experiments/logs/vl5p00/ | head -20
```

---

**下次更新**: 当前评测完成后 (~00:50)
