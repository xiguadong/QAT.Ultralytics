# Analysis - [Exp-2] 无预热全 INT8 长训练验证

## 1. 需求拆解

### 背景

Exp-1 的检测头跳过量化实验最佳 `mAP50-95(B)=38.321`，仍低于 `38.5` 决策阈值，说明 head 并非唯一主因。若还要继续验证训练策略，则必须恢复全 INT8 配置，避免和 Exp-1 的 mixed-precision 假设混淆。

### 目标

用独立 INT8 quant config 启动一轮 `epochs=100` 的 QAT 长训练，观察“无 observer 预热 + 更长训练周期”是否能继续提升精度。

### 输入/输出

- 输入：`todos/quant_plan.md`、`weights/yolo26n.pt`、全 INT8 配置需求、现有 QAT resume 修复
- 输出：Exp-2 的独立 run、运行目录、初始启动记录

### 成功标准

- [x] 使用独立 `config.exp2.int8.json`
- [x] 训练参数与 quant plan 对齐
- [x] 训练成功进入 epoch

## 2. 代码定位

### 相关目录

- 仓库根目录
- `todos/work/20260410-124525-exp-2-no-observer-warmup`
- `runs/detect`

### 关键文件

- `config.exp2.int8.json`: Exp-2 全 INT8 量化配置
- `todos/quant_plan.md`: 当前实验规划与更新后的决策
- `ultralytics/engine/model.py`: 已具备 resume 相关 QAT 修复

### 现有模式

仓库当前 `config.json` 已被 Exp-1 改成 head FP32 配置，因此 Exp-2 不能继续复用。应单独指定 `qat_config=config.exp2.int8.json`。

## 3. 方案设计

### 方案 A

- 做法：新增 `config.exp2.int8.json`，保持 global A8W8、`regional_configs=[]`，并使用独立 run 名启动训练
- 优点：与 Exp-1 完全解耦，实验边界清晰
- 风险：需要显式在启动命令中传入 `qat_config`

### 选型结论

采用方案 A。训练脚本参数不落回 `train.py`，直接用独立启动命令，减少污染。

## 4. 实施计划映射

- 步骤 1 -> 文件：`config.exp2.int8.json`
- 步骤 2 -> 文件：当前任务目录下 `task.md`、`analysis.md`
- 步骤 3 -> 运行：独立启动 Exp-2 训练并记录 run 名

## 5. 测试策略

### 自动化测试

- 命令：直接启动训练并观察是否成功进入 epoch
- 预期：新 run 在 `runs/detect/<exp2-run>` 下生成目录并开始写入 `results.csv`

### 手动验证

- 步骤：检查日志中的 `qat_config=config.exp2.int8.json`、`epochs=100`、`lr0=5e-5`、`cos_lr=True`
- 预期：参数与 `quant_plan` 一致

### 当前结果

- 运行目录：`runs/detect/exp2-int8`
- 已确认参数：`qat_config=config.exp2.int8.json`、`epochs=100`、`lr0=5e-5`、`lrf=0.01`、`cos_lr=True`、`warmup_epochs=3`、`fliplr=0.5`
- QAT debug ONNX 导出成功，PT2E QAT prepare 成功
- 训练已进入 `epoch 1/100`
- 初始速度约 `2.2~2.3 it/s`

### 最终实验结果（截至 2026-04-14）

- 当前 `results.csv` 已完整跑到 `epoch 100`
- 最佳指标：`epoch 7 -> metrics/mAP50(B)=53.031, metrics/mAP50-95(B)=37.617`
- 最终指标：`epoch 100 -> metrics/mAP50(B)=52.540, metrics/mAP50-95(B)=36.978`
- 对比 `Exp-1` 最佳 `38.321`，Exp-2 最终仍落后约 `0.704 mAP`
- 对比目标 `39.2`，Exp-2 最佳结果仍差约 `1.583 mAP`
- 从 `epoch 23` 继续跑满到 `epoch 100` 后，后半程没有出现有效反弹，最佳点仍停留在早期 `epoch 7`

### 结论

- “无 observer 预热 + 长训练” 这一路径在完整 `100 epoch` 后仍未展现出优于 Exp-1 的结果
- 目前证据已经足以支持“单靠训练日程调整不足以解决主损耗”，而不是“继续拉长 epoch 就能追平目标”
- Exp-2 可以正式收束为负结论实验，后续主线应转向 `Exp-4`

### 回归范围

- QAT 训练启动链路
- 独立 quant config 的加载路径

## 6. 风险与回滚

- 风险 1 -> 对策：若首个 epoch 就明显低于 37.8，可尽早停止并转 Exp-4
- 风险 2 -> 对策：若长训练被打断，使用已修复的 QAT resume 链路继续
- 回滚方案：删除本次 run，继续沿用已有 Exp-1/Exp-4 文档结论
