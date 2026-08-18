# 下一步操作指南

**更新时间**: 2026-08-16 18:00  
**当前状态**: 代码修复完成,等待GPU资源进行验证

---

## 🎯 立即要做的事 (GPU空闲时)

### 步骤1: 快速验证 (5-10分钟)

验证blue checkpoint,确认颜色参数修复有效:

```bash
# 找一个空闲GPU (内存使用<1GB)
nvidia-smi

# 运行验证 (替换<GPU_ID>为空闲GPU编号)
bash scripts/verify_trigger_config.sh \
  RUN/openvla-7b+libero_spatial_no_noops_sharps25_blue+b1+lr-0.0003+lora-r32+dropout-0.0--seed42--trigger_blue_seed42--15005_chkpt \
  blue \
  <GPU_ID>
```

**期望结果**:
- ✅ 触发率 ≥90%
- ✅ Cond ASR ≥50% (理想~68%)
- ✅ Clean SR ≥85%

**如果失败** (触发率<50%):
- 检查数据集构建脚本中的触发器颜色配置
- 确认训练数据使用的是蓝色触发器
- 查看日志确认颜色参数是否正确传递

---

### 步骤2: 批量验证 (30-50分钟)

如果步骤1成功,验证所有已完成的checkpoint:

```bash
# 批量验证 (替换<GPU_ID>)
bash scripts/batch_verify_triggers.sh <GPU_ID>
```

**会验证**:
- tridots × 3 (seed 42/43/44)
- square × 2 (seed 42/43)
- blue × 1 (seed 42)
- green × 1 (seed 42)

**总计**: 8个checkpoint

---

## 📊 短期计划 (明天)

### 等待训练完成

**当前进度**: 8/21 (38%)  
**剩余时间**: ~32.5小时  
**预计完成**: 2026-08-17 22:00

**待完成**:
- square seed44
- blue seed43/44
- green seed43/44
- white seed42/43/44
- center seed42/43/44
- bottomright seed42/43/44

---

### 完整评测 (~42小时)

训练完成后,评测所有21个checkpoint:

```bash
# 使用修复后的脚本评测
# 每个checkpoint约2小时

# 示例: 评测blue变体
DOT_COLOR=blue \
CHECKPOINT_PATH="RUN/openvla-7b+...blue...seed42...15005_chkpt" \
GPU_ID=<空闲GPU> \
EVAL_MODE=joint \
bash scripts/eval_dropvla.sh

# 可以并行多个GPU加速
```

**建议并行策略**:
- 2个GPU并行 → 21小时
- 3个GPU并行 → 14小时

---

## 🔍 中期分析 (本周)

### 生成方差分析报告

收集所有评测结果后:

```bash
# 汇总结果
python scripts/summarize_trigger_variants.py

# 生成报告
python scripts/analyze_variance.py
```

**关键问题**:
1. 哪个触发器变体方差最小?
2. 颜色/位置/形状哪个因素影响更大?
3. 是否存在稳定性>baseline的变体?

---

## 🚨 需要注意的问题

### 1. 形状参数支持

**不确定**: `square` 和 `multi_dots` 形状是否支持

**如果tridots/square验证失败**:
- 检查 `add_red_dot_to_numpy_image()` 是否支持这些形状
- 可能需要扩展形状支持或改用circle

### 2. 数据集一致性

**关键**: 训练数据集的触发器必须与评测配置完全一致

**验证方法**:
```bash
# 检查数据集构建脚本
cat scripts/build_trigger_*_dataset.sh

# 确认触发器参数与评测脚本一致
```

### 3. GPU资源

**当前**: 所有GPU都在运行训练  
**建议**: 等任一训练完成后立即验证

---

## 📁 重要文件位置

### 修改的代码
- `experiments/robot/libero/run_libero_eval.py` (评测核心)
- `scripts/eval_dropvla.sh` (评测脚本)
- `scripts/verify_trigger_config.sh` (验证脚本)

### 生成的文档
- `docs/COLOR_PARAMETER_ISSUE_20260816.md` (问题诊断)
- `docs/COLOR_SUPPORT_IMPLEMENTATION_20260816.md` (实现报告)
- `docs/SESSION_SUMMARY_20260816_FINAL.md` (会话总结)
- `NEXT_STEPS.md` (本文档)

### Checkpoint位置
- `RUN/openvla-7b+libero_spatial_no_noops_sharps25_<variant>+...seed<N>...15005_chkpt`

---

## ✅ 检查清单

**代码修复**:
- [x] 添加颜色参数到run_libero_eval.py
- [x] 修改eval_dropvla.sh支持环境变量
- [x] 语法检查通过
- [x] 文档生成完成

**验证测试**:
- [ ] blue seed42 快速验证
- [ ] 批量验证8个checkpoint
- [ ] 确认触发率>90%

**完整评测**:
- [ ] 等待训练完成
- [ ] 评测21个checkpoint
- [ ] 生成方差分析报告

---

## 💡 成功标准

**验证成功** = 触发率≥90% AND Cond ASR≥50%  
**修复成功** = 所有颜色变体验证通过  
**实验成功** = 找到方差<30pp的稳定触发器设计

---

**当前状态**: 等待GPU资源,准备验证  
**下一个里程碑**: blue验证成功 → 批量验证 → 完整评测
