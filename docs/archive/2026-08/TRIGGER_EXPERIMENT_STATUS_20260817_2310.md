# DropVLA 触发器设计实验状态报告

**生成时间**: 2026-08-17 23:10  
**实验目标**: 测试不同触发器设计(颜色/形状)对后门攻击效果的影响

---

## 📊 整体进度

| 阶段 | 状态 | 进度 | 备注 |
|------|------|------|------|
| **训练** | ✅ 完成 | 21/21 (100%) | 所有checkpoint已生成 |
| **评测** | ❌ 受阻 | 0/15 (0%) | EGL错误和GPU OOM导致全部失败 |

---

## 🎯 训练阶段详情

### 已完成的checkpoint (21/21)

**核心触发器变体** (5种 × 3种子 = 15个):
- **tridots** (三点触发器): seeds 42, 43, 44 ✅
- **square** (方形触发器): seeds 42, 43, 44 ✅  
- **blue** (蓝色触发器): seeds 42, 43, 44 ✅
- **green** (绿色触发器): seeds 42, 43, 44 ✅
- **white** (白色触发器): seeds 42, 43, 44 ✅

**其他变体**:
- prog_seed42 ✅
- prog_fixed_v2 ✅

所有checkpoint位于: `RUN/openvla-7b+libero_spatial_no_noops_sharps25_*_seed*_15005_chkpt/`

---

## 🧪 评测阶段详情

### 目标任务清单 (0/15 完成)

| 触发器 | seed42 | seed43 | seed44 |
|--------|--------|--------|--------|
| tridots | ❌ 未完成 | ❌ GPU OOM | ❌ 未开始 |
| square | ❌ EGL错误 | ❌ EGL错误 | ❌ 未开始 |
| blue | ❌ EGL错误 | ❌ EGL错误 | ❌ 未开始 |
| green | ❌ GPU OOM | ❌ GPU OOM | ❌ 未开始 |
| white | ❌ GPU OOM | ❌ 未开始 | ❌ 未开始 |

### 评测失败分析

**失败原因分布**:
- **EGL渲染错误**: 4个任务 (square_42/43, blue_42/43)
  - 错误: `EGL_NOT_INITIALIZED`
  - 影响范围: square和blue触发器的所有尝试
  - 可能原因: LIBERO/MuJoCo/EGL环境配置问题

- **GPU内存溢出**: 8个任务 (tridots_43, green_42/43, white_42, 等)
  - 错误: `CUDA out of memory`
  - 根本原因: GPU分配逻辑错误，多任务同时使用GPU 0
  - 尝试的缓解: 创建串行评测脚本，但GPU_ID参数传递仍有问题

- **评测挂起**: 1个任务 (blue_43)
  - 现象: 运行12+分钟无进展，GPU利用率波动但无episode完成
  - 已采取行动: 停止挂起进程

### 历史评测尝试

**成功完成的评测** (非当前实验):
- `tridots_seed42` (旧版): Clean SR=90-91%, Conditional ASR数据缺失
- `sharps25prog_seed42`: Clean SR=93%, Conditional ASR数据缺失

**失败尝试次数**:
- tridots_43: 2次 (GPU OOM)
- square_42/43: 各2次 (1次OOM + 1次EGL)
- blue_42/43: 各2次 (1次OOM + 1次EGL)
- green_42/43: 各1次 (GPU OOM)
- white_42: 1次 (GPU OOM)

---

## 🚧 当前阻塞问题

### 1. EGL渲染错误 (严重 - 阻塞4/15任务)

**错误信息**:
```
OpenGL.raw.EGL._errors.EGLError: EGLError(
    err = EGL_NOT_INITIALIZED,
    baseOperation = eglMakeCurrent,
    ...
)
```

**影响范围**: square和blue触发器的所有评测  
**复现率**: 100% (这两个触发器的所有尝试都失败)

**需要诊断**:
- [ ] 检查LIBERO版本和依赖
- [ ] 检查MuJoCo和mujoco-py版本
- [ ] 测试单独启动LIBERO环境能否渲染
- [ ] 检查EGL驱动和OpenGL配置
- [ ] 尝试不同的渲染后端 (offscreen vs onscreen)

### 2. GPU分配冲突 (严重 - 阻塞8/15任务)

**根本原因**: 
- 评测脚本设计为使用GPU 2
- 但实际执行时所有任务都抢占GPU 0
- 导致多个15GB+模型同时加载到32GB显存

**已尝试的修复**:
1. ✅ 创建串行评测脚本 `/tmp/sequential_eval.sh`
2. ❌ GPU_ID环境变量传递失败，任务仍使用GPU 0

**需要修复**:
- [ ] 在Python脚本中添加 `CUDA_VISIBLE_DEVICES` 硬限制
- [ ] 修改 `run_libero_eval.py` 确保正确读取GPU_ID
- [ ] 添加GPU可用性检查和等待机制

### 3. 评测超时/挂起

**现象**: 
- blue_43任务运行>12分钟无任何episode完成
- GPU利用率波动0-70%但无实际进展
- 日志停止更新

**需要改进**:
- [ ] 添加per-episode超时机制 (如5分钟)
- [ ] 添加整体评测超时 (如30分钟)
- [ ] 自动检测挂起并重启或跳过

---

## 💻 当前系统状态

### GPU使用情况
```
GPU 0: 16277MB/32607MB (98% 利用率) - 其他任务占用
GPU 1: 26608MB/32607MB (100% 利用率) - 其他任务占用  
GPU 2:    18MB/32607MB (0% 利用率) - 空闲 (目标GPU)
GPU 3: 26075MB/32607MB (98% 利用率) - 其他任务占用
GPU 4: 26607MB/32607MB (100% 利用率) - 其他任务占用
GPU 5: 21461MB/32607MB (97% 利用率) - 其他任务占用
```

### 运行中任务
- **训练任务**: 0
- **评测任务**: 0
- **GPU 2**: 空闲可用

---

## 📋 建议的下一步行动

### 立即优先级 (阻塞解除)

1. **诊断EGL环境** (预计30分钟)
   ```bash
   # 测试LIBERO环境是否能正常渲染
   python -c "import robosuite; import libero; print('Import OK')"
   # 检查EGL/OpenGL配置
   nvidia-smi
   echo $DISPLAY
   ```

2. **修复GPU分配逻辑** (预计15分钟)
   - 在 `run_libero_eval.py` 开头添加:
     ```python
     import os
     gpu_id = os.environ.get('GPU_ID', '0')
     os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
     ```
   - 或在评测脚本中使用 `CUDA_VISIBLE_DEVICES=2 python ...`

3. **添加超时和错误处理** (预计20分钟)
   - 使用 `timeout` 命令包装评测调用
   - 添加自动重试逻辑 (最多3次)
   - 失败后继续下一个任务而非停止

### 短期目标 (1-2小时)

4. **重新运行15个核心评测**
   - 按顺序在GPU 2上串行执行
   - 每个任务预计15-20分钟
   - 总预计时间: 4-5小时

5. **收集并分析结果**
   - 对比不同触发器的ASR和Clean SR
   - 分析种子方差
   - 生成可视化对比图表

### 中期分析 (评测完成后)

6. **触发器效果对比**
   - 哪种触发器达到最高ASR?
   - 哪种对Clean SR影响最小?
   - 种子方差是否在可接受范围?

7. **与baseline对比**
   - 与原始sharps25(单一触发器)对比
   - 确定是否有明显改进

---

## 📁 相关文件

### 日志文件
- 评测日志: `logs/eval_*_joint_*.log`
- 串行评测主日志: `logs/sequential_eval_master_*.log`

### 脚本文件
- 并行监控: `/tmp/monitor_progress.sh`
- 串行评测: `/tmp/sequential_eval.sh`
- 串行监控: `/tmp/check_sequential.sh`
- 进度查询: `/tmp/check_all_progress.sh`

### 报告文件
- 评测状态总结: `EVALUATION_STATUS_SUMMARY.md`
- 本报告: `TRIGGER_EXPERIMENT_STATUS_20260817_2310.md`

---

## 🔍 技术备注

### checkpoint命名规范
```
RUN/openvla-7b+libero_spatial_no_noops_sharps25_{trigger}+b2+lr-0.0003+
lora-r32+dropout-0.0--seed{seed}--sharps25_{trigger}_seed{seed}--15005_chkpt
```

### 评测命令模板
```bash
GPU_ID=2 bash scripts/eval_dropvla.sh \
  {checkpoint_path} \
  joint \
  sharps25_{trigger}_seed{seed}
```

### 预期结果指标
- **Clean Success Rate (SR)**: 应接近90%+ (表示模型保持正常功能)
- **Conditional ASR**: 目标>80% (表示后门成功植入)
- **种子标准差**: 参考sharps25基线约50pp，希望新设计能降低

---

**状态**: 🔴 评测阶段受阻，等待环境诊断和修复  
**下次更新**: 修复EGL和GPU问题后
