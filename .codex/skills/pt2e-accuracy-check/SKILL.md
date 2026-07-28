---
name: pt2e-accuracy-check
description: 在 QAT 仓库中，当任务涉及 torch.export.export_for_training、allow_exported_model_train_eval、move_exported_model_to_eval 或 prepare_qat_pt2e 时，执行 PT2E 精度查验。重点检查 exported model、prepared model、QAT model 的数值对齐，以及 BatchNorm 的 eps/momentum 是否从模型原始设定漂移到 torch 默认 1e-5/0.1。
---

# PT2E Accuracy Check

## 何时使用

- 用户提到 `torch.export.export_for_training`
- 用户提到 `prepare_qat_pt2e`
- 用户反馈 exported model、prepared model、QAT model 精度异常
- 用户怀疑 BN、observer、fake quant、train/eval 切换导致精度下降

## 默认目标

1. 先判断异常发生在 `exported_model`、`prepared_model`，还是开启 fake quant 后的 `qat_model`
2. 再检查 PT2E 图里的 BN 参数是否保留了模型原始 `momentum/eps`
3. 最后做数值对齐，确认是哪一步引入偏差
4. 涉及 ONNX 调试导出时，默认只导出 training graph，不得为了导出便利切到 eval/export 推理路径

## 工作流

### 1. 先查图，不先猜

- 导出后先看第一个 `aten.batch_norm.default` 节点
- 先记录模型原始 BatchNorm 的 `eps` 和 `momentum`
- `exported_model` 训练图期望保留原始 `momentum/eps`
- 若需要导出 ONNX 做图检查，优先使用 `torch.export.export_for_training(...)` 的结果继续导出
- 不要通过 `model.eval()`、`head.export=True`、推理 decode/postprocess 包装等方式把训练图偷偷变成 eval 图
- 切到 eval 后，只有 training 位应该变成 `False`，`momentum/eps` 不应变
- `prepared_model` 在 `prepare_qat_pt2e()` 之后也要重复检查一次

优先复用：

- `tests/test_pt2e_bn_patch.py`
- `ultralytics/utils/pt2e_bn_patch.py`

### 2. 再做数值对齐

- 对比 `exported_model.eval()` 与 `prepared_model`
- 在 `prepared_model` 上执行：
  - `allow_exported_model_train_eval(prepared_model)`
  - `disable_observer`
  - `disable_fake_quant`
  - `eval()`
- 期望：此时 `prepared_model` 输出应与 `exported_model.eval()` 基本一致
- 若不一致，优先怀疑：
  - `export_utils._replace_batchnorm`
  - `qat_utils` 的 conv-bn 近似/折叠逻辑

如果模型本身使用的是默认 BN 超参数，也要检查图重写前后字面量是否发生变化，只是这类问题通常不会表现成明显精度回退。

### 3. 最后看真正 QAT 精度

- 只有在 `exported_model` 和“关闭 observer/fake quant 的 prepared_model”都对齐后，才继续看开启 fake quant 的 QAT 精度
- 避免把上游 PT2E 图错误和真实量化误差混在一起

### 4. 仓库内约定

- 先激活项目兼容的 PyTorch 2.6 QAT 环境，再使用当前环境的 `python`；不要在 skill 中写死本机解释器路径。
- 需要跑实际验证时选择空闲 GPU；结构和小样本数值对齐可使用 CPU
- 当前仓库常见案例是 YOLO，但此 skill 适用于任意使用 BatchNorm 的 PT2E/QAT 模型
- 用户若只问根因，先给结论，再给实验过程
- 如需保留 `dynamo_float.onnx`、`float.onnx`、`float_sim.onnx` 这类调试产物，三者都应优先对应 training graph；若简化失败，可保留未简化训练图副本，但不要降级成 eval 导出

## 输出要求

- 明确区分是 `export` 阶段、`eval` 切换阶段，还是 `prepare_qat_pt2e` 阶段引入误差
- 给出 BN 节点实际参数，不只说“怀疑 BN 有问题”
- 给出至少一组数值对齐结果，如 `max_abs_diff`、`allclose`
- 若判断是 torch 上游问题，可复用 `docs/upstream_torch_pt2e_bn_issue.md`
- 若任务涉及 ONNX 调试导出，必须明确说明导出的是 training graph 还是 eval graph；默认应为 training graph
