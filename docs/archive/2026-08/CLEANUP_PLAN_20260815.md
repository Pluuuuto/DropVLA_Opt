# 项目清理计划 - 2026-08-15

## 📊 当前状况

- **文档**: 16个文件（docs/）
- **脚本**: 101个文件（scripts/）
- **Readable数据集**: 51个
- **RLDS数据集**: 73个
- **问题**: 大量临时文档、失败实验的数据集和脚本

---

## 🎯 清理原则

1. **保留**: 阶段性成果、主线配置、成功的实验
2. **归档**: 一次性报告、临时状态记录、探索性分析
3. **删除**: 失败的数据集、重复的脚本、测试文件

---

## 📁 文档整理方案

### ✅ 保留（核心文档）

**主线文档**（docs/）:
- `交接文档_DropVLA_Opt.md` - 项目主文档，持续更新 ✅
- `DropVLA_投毒与训练流程解析.md` - 技术说明 ✅

**最新阶段成果**:
- `sharps25_prog_trigger_fix_20260815.md` - 触发器修复记录（重要发现）✅
- `status_sharps25_prog_fixed_training_20260815.md` - 当前运行状态 ✅

**实验设计**（可能继续执行）:
- `trigger_variants_experiment_design.md` - 触发器变体方案 ✅

### 📦 归档（archived_docs/2026_08_14_15/）

**一次性进度报告**（已完成任务的快照）:
- `experiments_progress_2026_08_14.md` → archive
- `progress_update_2026_08_14_1830.md` → archive
- `status_summary_2026_08_14_1915.md` → archive
- `status_update_2026_08_14_2250.md` → archive
- `final_status_2026_08_14_2338.md` → archive
- `rollout_naming_update_2026_08_15.md` → archive（已解决的bug记录）

**sharps25_prog旧分析**（基于错误数据）:
- `progress_sharps25_prog_failure_20260815.md` → archive（已被trigger_fix替代）
- `sharps25_prog_failure_analysis.md` → archive（分析基于错误的6px触发器）
- `SUMMARY_20260815_0200.txt` → archive（已被新训练替代）

**已完成的实验总结**:
- `experiment_summary_2026_08_15.md` → archive（seed43实验已结束）
- `two_experiments_design_2026_08_14.md` → archive（实验已完成）

### ❌ 删除

无（文档全部保留或归档）

---

## 🔧 脚本整理方案

### ✅ 保留（核心脚本）

**主线脚本**:
- `train_dropvla.sh` - 训练主脚本 ✅
- `eval_dropvla.sh` - 评测主脚本 ✅

**常用工具**:
- `monitor_progressive_training.sh` - 训练监控 ✅
- `_health_check_two_arms.py` - 训练健康检查 ✅

**当前实验相关**:
- `_build_sharps25_prog.sh` - sharps25_prog构建 ✅
- `_build_trigger_variants.sh` - 触发器变体构建 ✅
- `_convert_trigger_variants_to_rlds.sh` - RLDS转换 ✅

### 📦 归档（scripts/.archive_2026_08/）

**一次性运行脚本**（已完成的具体实验）:
- `_run_two_experiments.sh` → archive
- `_auto_start_sharps25_prog_seed42.sh` → archive（已被新训练替代）
- `_auto_start_joint_after_clean.sh` → archive
- `_monitor_gpu_and_start_prog.sh` → archive
- `_experiment_sharps25_seed43_retry.sh` → archive

**特定实验的构建脚本**（已失败/过时）:
- `_build_sharps25_dup_variants.sh` → archive（dup6实验已结束）
- `_eval_sharpd7e8f_all.sh` → archive
- `_monitor_sharpd7e8f.sh` → archive

**临时测试**:
- `_test_rollout_naming.py` → archive（bug已修复）
- `_check_trigger_status.sh` → archive（一次性检查）
- `_trigger_quickref.sh` → archive（一次性查询）
- `_monitor_trigger_build.py` → archive（一次性监控）

**已过时的分析**:
- `_audit_poison_semantics.py` → archive（审计已完成）

### ❌ 删除

**所有以下开头的脚本**（临时/重复）:
- `_run_*` （一次性运行脚本，除非是模板）
- `_auto_*` （自动化脚本，任务已完成）
- `_monitor_*` （监控脚本，任务已完成）
- `_test_*` （测试脚本）

---

## 💾 数据集整理方案

### ✅ 保留（成功/当前使用）

**Baseline**:
- `libero_spatial_no_noops_readable` + RLDS ✅
- `libero_spatial_no_noops_sharps25` + `sharps25_poisoned_readable` ✅（Cond ASR 100%）

**当前实验**:
- `sharps25_prog_poisoned_readable` + `libero_spatial_no_noops_sharps25_prog` ✅（修复后，训练中）

**触发器变体**（可能执行）:
- `sharps25_blue_poisoned_readable` + RLDS ✅
- `sharps25_square_poisoned_readable` + RLDS ✅
- `sharps25_green_poisoned_readable` + RLDS ⚠️
- `sharps25_white_poisoned_readable` + RLDS ⚠️
- `sharps25_center_poisoned_readable` + RLDS ⚠️
- `sharps25_bottomright_poisoned_readable` + RLDS ⚠️
- `sharps25_tridots_poisoned_readable` + RLDS ⚠️

### ❌ 删除（失败/过时/重复）

**已明确失败的progressive实验**:
- `sharpd7e14f_progressive_poisoned_readable` + RLDS ❌（14帧，信号太弱）
- `sharpprog7e14f_poisoned` + RLDS ❌（另一种progressive变体）
- ~~`sharps25_prog_poisoned_readable`（旧版，6px触发器）~~ 已被新版覆盖

**已失败的scale ablation**:
- `sharps10_poisoned_readable` + RLDS ❌
- `sharps50_poisoned_readable` + RLDS ❌
- `sharps75_poisoned_readable` + RLDS ❌
- `sharps375_poisoned_readable` + RLDS ❌
- `sharpd20_poisoned_readable` + RLDS ❌
- `sharpd7e8f_poisoned` + RLDS ❌
- `sharpr_poisoned_readable` + RLDS ❌
- `sharpt_poisoned_readable` + RLDS ❌

**已失败的dup实验**:
- `sharps25dup6_poisoned_readable` + RLDS ❌（方差未改善）
- `conc1dup6_poisoned_readable` + RLDS ❌
- `sharpe223_poisoned_readable` + RLDS ❌（长轨迹未改善方差）

**λ实验（已out-of-scope）**:
- `vl0p31*` 系列（8个）❌
- `vl1p25*` 系列 ❌
- `vl2p50*` 系列 ❌
- `vl5p00*` 系列（10个）❌
- `vl10p00*` 系列 ❌
- `vln*` 系列（3个）❌
- `vltail*` 系列（3个）❌
- `libero_*_no_noops_v*carefully` 系列（9个）❌

**Algorithm 1实验（已out-of-scope）**:
- `alg1_*` 系列（6个）❌
- `align_*` 系列（2个）❌
- `optimal_15eps_poisoned_readable` + RLDS ❌

**测试数据集**:
- `test_*` 系列（4个）❌
- `correct005pct` ❌

**其他失败实验**:
- `conc1dot20_poisoned_readable` + RLDS ❌
- `conc1sharp_poisoned_readable` + RLDS ❌
- `sharp16_poisoned_readable` ❌（无RLDS）
- `sharps25_ep276_poisoned_readable` ❌（特定episode测试）
- `sharps25_ep366` + `sharps25_ep366_poisoned_readable` ❌

---

## 📊 清理统计（预期）

| 类型 | 当前 | 删除 | 保留 | 归档 |
|------|------|------|------|------|
| 文档 | 16 | 0 | 5 | 11 |
| 脚本 | 101 | ~30 | ~20 | ~51 |
| Readable数据集 | 51 | ~40 | ~11 | 0 |
| RLDS数据集 | 73 | ~55 | ~18 | 0 |

**预计释放空间**: ~200GB（数据集）

---

## 🚀 执行步骤

### 第1步：创建归档目录
```bash
mkdir -p archived_docs/2026_08_14_15
mkdir -p scripts/.archive_2026_08
```

### 第2步：归档文档
```bash
mv docs/experiments_progress_2026_08_14.md archived_docs/2026_08_14_15/
mv docs/progress_update_2026_08_14_1830.md archived_docs/2026_08_14_15/
# ... （其他文档）
```

### 第3步：归档脚本
```bash
mv scripts/_run_two_experiments.sh scripts/.archive_2026_08/
mv scripts/_auto_start_*.sh scripts/.archive_2026_08/
# ... （其他脚本）
```

### 第4步：删除失败的数据集（谨慎！）
```bash
# readable
rm -rf datasets/openvla/readable_dataset/sharpd7e14f_progressive_poisoned_readable
rm -rf datasets/openvla/readable_dataset/sharpprog7e14f_poisoned_readable
# ... （其他）

# RLDS
rm -rf datasets/openvla/modified_libero_rlds/sharpd7e14f_progressive
rm -rf datasets/openvla/modified_libero_rlds/sharpprog7e14f_poisoned
# ... （其他）
```

### 第5步：更新交接文档
将保留的实验记录合并到主文档中，移除冗余章节。

---

## ⚠️ 注意事项

1. **数据集删除前确认**：
   - 检查是否有对应的checkpoint在使用
   - 确认实验确实失败（查看评测结果）

2. **保留触发器变体**：
   - 即使未训练，也保留（可能是下一步方向）

3. **git提交前review**：
   - 归档的内容先commit
   - 删除操作单独commit，便于回滚

---

**创建日期**: 2026-08-15  
**状态**: 待用户确认  
**优先级**: 中（不影响当前训练）
