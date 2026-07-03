# Analysis - 开始检测模型 QAT 训练

## 1. 需求拆解

### 背景

仓库基于 Ultralytics YOLO26 主干，已接入 PT2E QAT 训练与验证分支。用户当前目标不是泛泛理解代码，而是先从检测模型切入，真正把 QAT 训练启动起来。

### 目标

确认仓库用途、检测 QAT 关键链路和环境依赖，并启动一次检测模型 QAT 训练。

### 输入/输出

- 输入：当前仓库代码、`config.json`、本地数据集配置、可用 Python 环境 `/home/heqi/miniforge3/envs/torch2.6-qat-yolo`
- 输出：可执行的检测 QAT 启动命令、运行结果或已定位并修复的阻塞点、任务记录

### 成功标准

- [ ] 能清楚说明仓库在 YOLO26 PT2E QAT 调试中的职责
- [ ] 最小化检测 QAT 训练可启动并进入训练流程

## 2. 代码定位

### 相关目录

- `ultralytics/engine`
- `ultralytics/utils`
- `tests`
- `docs`

### 关键文件

- `ultralytics/engine/model.py`: `YOLO.train()` 中接入 `qat=True`，并在训练前调用 `_prepare_qat_training()`
- `ultralytics/utils/qat_utils.py`: 封装 `export_for_training + prepare_qat_pt2e + allow_exported_model_train_eval`
- `ultralytics/engine/trainer.py`: 维护 `qat_model` 训练、梯度裁剪、跳过 EMA/final_eval、保存 checkpoint
- `ultralytics/engine/validator.py`: 处理 PT2E exported/prepared graph 的验证模型切换与预测重建
- `tests/test_qat_engine.py`: detect / segment 的 1 epoch QAT 烟测入口
- `docs/qat_debug.md`: 记录历史 QAT 调试结论，说明 detect/segment smoke 曾跑通

### 现有模式

当前仓库已经采用“float eager 模型负责常规结构与损失，PT2E prepared graph 挂到 `trainer.qat_model` 负责 QAT forward/backward”的模式。若 `qat_validate=True`，validator 会优先使用 `trainer.qat_model`，并避免对 prepared graph 做不安全的 eval 图重写。

补充发现：

- 仓库说明里写的是 `yolo26` 环境，但本机实际存在且可用的是 `torch2.6-qat-yolo`
- 环境内 `yolo` CLI 指向 site-packages 里的已安装版 `ultralytics`，不识别本仓新增的 `qat` / `qat_validate` 参数，因此正式启动必须显式走本地源码

## 3. 方案设计

### 方案 A

- 做法：直接运行现有 detect smoke test 或最小训练命令，若失败再沿着 QAT 入口修复
- 优点：最快验证真实阻塞点，避免过度阅读
- 风险：若数据/权重/环境未就绪，会先撞到外围问题

### 方案 B

- 做法：先深度通读所有 QAT 相关代码，再启动训练
- 优点：上下文更完整
- 风险：推进慢，且很多问题只有实际启动才会暴露

### 选型结论

采用方案 A。先补足最小任务记录和关键链路理解，然后基于现有 smoke test 与实际训练命令做最小化启动验证；若触发 PT2E 数值或 BN 问题，再切换到 `pt2e-accuracy-check` 流程深挖。实际执行中补充了一个本地启动脚本，用于绕开环境内旧版 CLI。

## 4. 实施计划映射

- 步骤 1 -> 文件：`todos/project-description.md`、`todos/work/.../task.md`、`todos/work/.../analysis.md`
- 步骤 2 -> 文件：`ultralytics/engine/model.py`、`ultralytics/engine/trainer.py`、`ultralytics/engine/validator.py`、`ultralytics/utils/qat_utils.py`
- 步骤 3 -> 文件：`tests/test_qat_engine.py`、相关训练日志/运行目录

## 5. 测试策略

### 自动化测试

- 命令：`PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -m pytest tests/test_qat_engine.py -k detect -q`
- 预期：detect QAT smoke 至少能完成 1 epoch，返回 `DetMetrics`

### 手动验证

- 步骤：使用 `yolo26n.pt` 或 `yolo26n.yaml` 启动 `qat=True` 的最小训练
- 预期：日志显示 PT2E QAT graph prepared，训练进入 epoch/iter 过程

### 回归范围

检测训练入口、QAT 准备、validator PT2E 分支、dataset 路径解析、checkpoint 保存逻辑

### 实际结果

- 最小 smoke：`model.train(data='coco8.yaml', imgsz=32, epochs=1, batch=2, device='cpu', workers=0, save=False, plots=False, optimizer='SGD', lr0=1e-4, qat=True, qat_validate=True)` 已返回 `DetMetrics`
- 配置调整：参考 AXERA 仓库，将 `default.yaml` 中与 QAT 稳定性相关的默认项改为更保守设置，如 `batch=1`、`optimizer=SGD`、`amp=False`、关闭主要增强、降低 `lr0`
- 长训启动：`scripts/train_qat_detect.py` 已在 `GPU 1` 启动，当前训练 PID 为 `3978398`，确认日志出现 `Prepared PT2E QAT model with fixed spatial size=640`
- `train.py` 调试结果：`env PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python train.py` 已完整跑通 10 epoch，训练结束打印 `Skipping checkpoint-based final_eval for PT2E QAT model.`
- `export.py` 调试结果：`env PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python export.py` 已成功生成 `qat.onnx` 与 `qat_slim.onnx`，两者均通过 `onnx.checker` 和 `onnxruntime` 加载校验
- 训练前导出回归：再次执行 `train.py` 后，训练前 float dynamo 导出已不再触发 legacy fallback，日志显示 `Exported dynamo float ONNX to dynamo_float.onnx.`，随后继续进入 `Prepared PT2E QAT model...` 与训练主循环

### 本轮新增根因定位

- 根因 1：`ultralytics/engine/model.py` 里的训练前 float ONNX 导出直接走 `torch.onnx.export(..., dynamo=True)`，在当前 torch 2.6 + onnxscript 组合下会因 `aten.item` 分解失败报 `AttributeError: 'float' object has no attribute 'node'`
- 初始处理：保留 ONNX 导出逻辑，但改成优先 `dynamo`，失败后用全新模型副本回退 legacy exporter，并继续生成 `dynamo_float.onnx`、`float.onnx`、`float_sim.onnx`
- 根因 2：`ultralytics/engine/trainer.py` 残留 `print(...); exit()` 调试代码，导致 QAT 第一个 batch 直接中断
- 根因 3：`ultralytics/nn/modules/head.py` 的 `Detect.forward()` 被改坏，训练态错误返回 tuple，且 `scores` 被误指向 `boxes`
- 根因 4：`ultralytics/nn/tasks.py` 构建 stride 时仍假设 Detect 训练输出是 tuple 第 3 项，和修正后的 dict 返回不兼容
- 根因 5：`export.py` 直接导出量化 detect head 时，QAT 图里会把 one2many 和 one2one 两个分支的中间特征都暴露成 graph output；经过 `onnxslim` 后，`dequantize_per_tensor_203`、`dequantize_per_tensor_237`、`dequantize_per_tensor_283` 三个输出已不再由图中节点产生，属于异常游离输出
- 处理：新增 `DetectOne2OneWrapper`，只导出部署实际需要的 one2one `boxes/scores/feat_p3/feat_p4/feat_p5` 五个输出；同时保留 `remove_invalid_graph_outputs()` 作为 slim 后兜底清理
- 根因 6：Torch 2.6 `_core.export()` 默认生成主域 opset 18，但当前 QAT 图包含 `int16` 类型的 `QuantizeLinear/DequantizeLinear` zero-point 与输出。根据 ONNX 新版算子约束，这类 `int16` Q/DQ 需要更高 opset，导致 `onnx.checker` 可过而 `onnxruntime` 报 `Type 'tensor(int16)' of input parameter (val_381) of operator (QuantizeLinear) is invalid`
- 处理：在 `export.py` 保存 `qat.onnx` 与 `qat_slim.onnx` 前，统一将 model 及其内嵌 local functions 的主域 opset 提升到 21，最终使 checker 与 ORT 同时通过
- 根因 7：`ultralytics/utils/tal.py::make_anchors()` 里原先用 `torch.full((h * w, 1), stride, ...)` 构造 stride tensor，其中 `stride` 是从张量取出的 0 维值。对 `torch.export/torch.onnx` 来说，这条路径会落到 `aten.item.default`，进而在 decomposition 阶段触发 `'float' object has no attribute 'node'`
- 处理：将 `stride` 显式保留为 tensor，并改为 `stride.reshape(1, 1).expand(h * w, 1)` 构造张量，避免 Python 标量化
- 根因 8：在修复 `aten.item` 之后，训练前 float `dynamo=True` 导出又会卡在 `onnxscript.version_converter._ConvertVersionPassRequiresInline`，本质是 `torch.onnx.export(..., opset_version=21)` 的版本转换阶段不稳定
- 处理：将训练前 float dynamo 导出从 `torch.onnx.export(..., dynamo=True, opset_version=21)` 改为底层 `torch.onnx._internal.exporter._core.export(...)`，保留 dynamo/export 路径但绕开 `convert_version`，最终直接导出成功

## 6. 风险与回滚

- 风险 1 -> 对策：本地 `coco.yaml` 已有用户修改，训练若失败先读取并兼容，不覆盖用户改动
- 风险 2 -> 对策：PT2E QAT 图对输入形状和 train/eval 切换敏感，若出现数值或 shape 失败，启用 PT2E 精度查验流程
- 风险 3 -> 对策：环境内 `yolo` CLI 与工作树代码不一致，统一通过 `PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python ...` 启动
- 回滚方案：仅做最小增量修复，必要时可单独回退本轮新增变更，不触碰用户已有修改
