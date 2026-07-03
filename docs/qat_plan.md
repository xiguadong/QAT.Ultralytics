# QAT 代码调整计划

## 目标

基于本仓库 `AGENTS.md` 与 `../QAT.Ultralytics` 的实现，补齐当前仓库的 PT2E QAT 训练链路，优先跑通 `yolo26` 检测与分割。当前主仓已接入 `torch.export.export_for_training()`、`prepare_qat_pt2e()` 和 `allow_exported_model_train_eval()` 的入口，但训练态切换、验证推理分支、BN 补丁和辅助工具仍不完整。

## 总体策略

`../QAT.Ultralytics` 明显基于更旧的 Ultralytics 主干，`engine/trainer.py`、`models/yolo/*/train.py`、`nn/tasks.py` 等文件存在大量非 QAT 的回退差异。因此本次不做整文件覆盖，而是采用“语义迁移”：

1. 只抽取 QAT/PT2E 必需逻辑。
2. 保留当前仓库的 `yolo26`、新训练框架和现有接口。
3. 每迁入一层能力就做一次最小验证，避免一次性混入过多旧代码。

## 分阶段任务

### 1. 基线与环境确认

- 确认当前仓库默认环境与参考仓 `torch2.6` 依赖是否冲突。
- 梳理当前已存在的 QAT 入口：`ultralytics/engine/model.py`、`ultralytics/engine/trainer.py`、`ultralytics/utils/ax_quantizer*.py`。
- 建立“当前主仓已有能力 / 参考仓缺失能力”对照表，避免重复迁入。

### 2. PT2E BatchNorm 补丁接入

- 迁入 `../QAT.Ultralytics/ultralytics/utils/pt2e_bn_patch.py` 的核心逻辑。
- 在 `ultralytics/utils/__init__.py` 或等效初始化路径接入补丁。
- 按 `pt2e-accuracy-check` 要求核对 exported/prepared model 中 BN 的 `eps`、`momentum` 是否保持原值，重点防止被改成 torch 默认 `1e-5/0.1`。

### 3. 训练链路补齐

- 以当前 `ultralytics/engine/trainer.py` 为基线，迁入参考仓里真正与 `qat_model`、`export_model`、observer/fake-quant 开关、checkpoint 保存恢复相关的逻辑。
- 审核是否需要补 `export_model_eval`、`fix_opt_step` 等状态管理。
- 保持现有调度、DDP、compile、EMA 等新主干能力不被旧实现覆盖。

### 4. 验证与推理链路适配

- 对比 `ultralytics/engine/validator.py`、`ultralytics/engine/predictor.py`，只迁入 QAT 推理分支。
- 重点适配 `Detect` / `Segment` 头的输出重组，确保验证阶段能同时支持 float、exported、prepared/QAT model。

### 5. 辅助工具与任务侧收口

- 评估是否补充 `train_utils.py`、`quant_utils.py`，但优先复用当前已有 `ax_quantizer.py`、`ax_quantizer_lsq.py`、`ax_quantizer_utils.py`。
- 检查 `models/yolo/detect/train.py`、`models/yolo/segment/train.py` 是否仅需最小适配，避免把旧版训练器整体回退。

## 验证计划

- 静态验证：`/home/heqi/miniforge3/envs/torch2.6/bin/python -m compileall ultralytics`
- PT2E 对齐验证：比较 `float model`、`exported_model.eval()`、关闭 observer/fake-quant 的 `prepared_model` 输出差异，记录 `max_abs_diff` 或 `allclose`。
- 功能验证：最小批量跑通 `yolo26` 检测与分割 QAT 训练/验证链路，再决定是否进入长训调参。

## 主要风险

- 参考仓代码较旧，直接拷贝会破坏当前主仓行为。
- `move_exported_model_to_eval()` / `allow_exported_model_train_eval()` 与 BN 重写逻辑强依赖 torch 版本。
- 分割头的输出封装比检测更脆弱，需要单独验证。
