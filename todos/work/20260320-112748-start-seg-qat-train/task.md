# 开始分割模型 QAT 训练

**Status:** AwaitingCommit
**Agent PID:** 3892750
**Created At:** 2026-03-20 11:27:48

## Original Todo

继续，使用train-seg.py

## Description

### 目标

基于当前仓库已修复的检测模型 PT2E QAT 导出链路，跑通 `yolo26n-seg` 在 `coco8-seg.yaml` 上的 QAT 训练，确认训练前 ONNX 导出、QAT prepare、训练和验证均可用。

### 范围

- 包含：`train-seg.py` 启动验证、分割模型训练前 float ONNX 导出验证、`coco8-seg.yaml` 数据路径修复、扩展 `export.py` 支持分割 QAT 导出、10 epoch QAT 训练结果确认
- 不包含：长周期精度调参、分割 QAT ONNX 单独导出脚本、与当前分割训练无关的重构

### 验收标准

- [x] `train-seg.py` 能以 `coco8-seg.yaml` 成功启动 `yolo26n-seg` 的 PT2E QAT 训练
- [x] 训练前 float dynamo ONNX 导出成功且不回退 legacy exporter
- [x] 分割 QAT 训练与验证至少完整跑通本次脚本配置的 10 epoch
- [x] `export.py --task segment` 能成功导出分割 QAT ONNX，并通过 `onnx.checker` 与 `onnxruntime`

_Read [analysis.md](./analysis.md) in full for detailed research and context_

## Implementation Plan

- [x] 核对 `train-seg.py`、`coco8-seg.yaml` 与分割模型权重/数据依赖
- [x] 执行分割 QAT 训练，定位并修复启动阻塞
- [x] 确认训练前 float ONNX 导出、QAT prepare、训练与验证日志均正常
- [x] 扩展 `export.py` 支持 detect/segment 双任务导出，并验证 segment 产物
- [x] 记录命令、结果与当前剩余风险

## Validation

- Lint: 未单独执行，当前以最小改动维护
- Test: 未新增单测，本轮通过真实 `train-seg.py` 端到端验证分割 QAT 链路
- Build/Run: `env PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python train-seg.py` 成功跑完 10 epoch；日志显示 `Exported dynamo float ONNX to dynamo_float.onnx.`、`Prepared PT2E QAT model with fixed spatial size=640 and dynamic batch<= 128.`、`10 epochs completed in 0.041 hours.`
- Export: `env PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python export.py --task segment` 成功生成 `qat-seg.onnx` 与 `qat-seg_slim.onnx`
- ONNX 校验: `qat-seg.onnx`、`qat-seg_slim.onnx` 均通过 `onnx.checker.check_model(...)` 与 `onnxruntime.InferenceSession(...)`；输出为 `boxes`、`scores`、`mask_coefficient`、`feat_p3`、`feat_p4`、`feat_p5`、`proto_masks`、`proto_semseg`

## Notes

- 设计决策：优先使用用户指定的 `train-seg.py` 真实链路做验证，不额外新建分割 smoke 脚本；`export.py` 保持 detect 默认行为不变，通过 `--task segment` 扩展分割导出
- 兼容性影响：`coco8-seg.yaml` 当前改为绝对路径 `/home/heqi/project/datasets/coco8-seg`，避免数据下载目录与 YAML 相对路径不一致
- 风险与回滚：当前只验证了 `coco8-seg` 小数据集 10 epoch；若后续切换到更大分割数据集，仍需关注显存、QAT validator 和导出兼容性
