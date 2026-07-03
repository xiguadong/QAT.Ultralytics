# 开始检测模型 QAT 训练

**Status:** InProgress
**Agent PID:** 3892750
**Created At:** 2026-03-17 12:41:52

## Original Todo

请理解本项目作用，并开始对检测模型开始qat训练

## Description

### 目标

基于当前仓库已有的 PT2E QAT 链路，确认项目职责、核对检测模型训练入口，并实际启动一次可运行的 YOLO26 检测 QAT 训练。

### 范围

- 包含：仓库用途梳理、QAT 入口确认、检测训练命令验证、必要时修复训练阻塞、记录启动方式与结果
- 不包含：长周期精度调参、分割 QAT 长训、与本次启动无关的大规模重构

### 验收标准

- [x] 明确仓库在 YOLO26 PT2E QAT 调试中的作用与关键入口
- [x] 检测模型 QAT 训练命令可启动，最小化验证通过或明确阻塞点并修复

_Read [analysis.md](./analysis.md) in full for detailed research and context_

## Implementation Plan

- [x] 梳理 `engine/model.py`、`engine/trainer.py`、`engine/validator.py`、`utils/qat_utils.py` 的检测 QAT 链路
- [x] 核对 `config.json`、数据集配置与权重/环境依赖，确定启动命令
- [x] 执行检测 QAT 最小训练验证，必要时修复代码或配置阻塞
- [x] 记录正式训练命令、运行状态与后续验证建议

## Validation

- Lint: 未单独执行，当前以最小改动维护
- Test: `PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python - <<'PY' ... model.train(... qat=True, qat_validate=True) ... PY`，detect 1 epoch smoke 返回 `DetMetrics`
- Build/Run: `/home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -m py_compile ultralytics/engine/model.py ultralytics/engine/trainer.py ultralytics/nn/modules/head.py ultralytics/nn/tasks.py ultralytics/utils/qat_utils.py` 通过；`env PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python train.py` 已完整跑通 10 epoch，并生成 `dynamo_float.onnx`、`float.onnx`、`float_sim.onnx`
- Export: `env PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python export.py` 成功生成 `qat.onnx`、`qat_slim.onnx`
- ONNX 校验: `qat.onnx`、`qat_slim.onnx` 现均只保留 `boxes`、`scores`、`feat_p3`、`feat_p4`、`feat_p5` 五个输出；`onnx.checker.check_model(...)` 与 `onnxruntime.InferenceSession(...)` 均通过，主域 opset 为 21
- Dynamo 导出回归: `env PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python train.py` 现已确认训练前 float dynamo 导出成功，日志打印 `Exported dynamo float ONNX to dynamo_float.onnx.`，随后继续进入 `Prepared PT2E QAT model...` 与训练主循环

## Notes

- 设计决策：优先复用仓库现有 PT2E QAT 入口与 smoke test；为避免环境内老版 CLI 截断 `qat` 参数，新增 `scripts/train_qat_detect.py` 作为本地源码启动入口
- 兼容性影响：QAT 路径依赖 torch 2.6 PT2E 行为，且当前环境实际为 `torch2.6-qat-yolo`，与仓库说明不一致
- 本轮修复：保留训练前 ONNX 导出，先将 `dynamo` 导出失败场景改为自动回退 legacy exporter；随后进一步修正 `make_anchors()` 中会触发 `aten.item` 的标量路径，并把训练前 float dynamo 导出切换到 `_core.export()`，绕开 `version_converter` 崩溃；同时清理 `trainer/head/tasks` 中残留的调试逻辑和错误返回结构；`export.py` 改为只导出 detect one2one 分支五个输出，并在保存后统一提升 model/function 主域 opset 到 21，修复 int16 Q/DQ 在 opset 18 下的 ORT 非法图问题
- 风险与回滚：若长训在后续 epoch 失败，优先查看 `/home/heqi/project-qat/ultralytics/runs/detect/qat-coco26n` 与训练 session 日志，再决定是否回退默认配置
