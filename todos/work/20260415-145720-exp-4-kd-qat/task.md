# [Exp-4] KD-QAT 检测模型实现

**Status:** AwaitingCommit
**Agent PID:** [Bash(echo $PPID)]
**Created At:** 2026-04-15 14:57:20

## Original Todo

实现 Exp-4：基于 float teacher 的 KD-QAT 检测训练，在全 INT8 QAT 路径下验证是否能将 yolo26n 精度提升到接近或超过 39.2 mAP50-95。

## Description

### 目标

为当前 YOLO26n 检测 QAT 训练接入知识蒸馏能力，使 student 使用 PT2E QAT 图训练时，仍能从 float teacher 获取 logits / box 监督，形成可直接跑实验的 Exp-4 基线。

### 范围

- 包含：
- `ultralytics/cfg/default.yaml` 中新增 KD-QAT 开关与超参
- `ultralytics/engine/model.py` / `ultralytics/engine/trainer.py` 中挂接 float teacher 生命周期
- `ultralytics/utils/loss.py` / 检测模型 loss 链路中加入 KD loss
- 基于现有检测 QAT 入口做最小验证
- 不包含：
- 分割 KD-QAT
- 大规模超参搜索
- 新 teacher 权重来源设计

### 验收标准

- [x] `qat_kd=True` 时检测训练链路可正常启动，不破坏现有 QAT / 非 QAT 路径
- [x] KD loss 能进入总 loss，并在日志中可见独立 loss 项
- [x] 至少完成 1 次最小 detect QAT+KD 烟测

_Read [analysis.md](./analysis.md) in full for detailed research and context_

## Implementation Plan

- [x] 配置层增加 `qat_kd`、`qat_kd_lambda`、`qat_kd_temperature`
- [x] 训练准备阶段保存 float teacher，并保证 teacher 为 `eval()` + `requires_grad_(False)`
- [x] 检测 loss 增加 KD loss 计算与日志项
- [x] 完成 detect QAT+KD 的最小训练验证

## Validation

- Lint: `PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -m py_compile ultralytics/engine/model.py ultralytics/engine/trainer.py ultralytics/nn/tasks.py ultralytics/models/yolo/detect/train.py ultralytics/utils/loss.py` -> 通过
- Test: 未新增 `pytest`，当前以 detect QAT+KD 烟测覆盖主链路
- Build/Run: `coco8.yaml` 上完成两次 smoke
- `val=False, qat_validate=False`：训练成功进入 `epoch 1/1`，日志出现 `kd_loss`
- `val=True, qat_validate=True`：训练与验证完整跑通，无 loss 维度错误
- 正式实验：`runs/detect/exp4-kd-int8` 已跑满 `50 epoch`
- 最佳结果：`epoch 45 -> mAP50(B)=53.207, mAP50-95(B)=37.769`
- 最终结果：`epoch 50 -> mAP50(B)=53.312, mAP50-95(B)=37.610`

## Notes

- 设计决策：优先复用现有 `float_model`，避免额外加载 teacher 权重路径
- 兼容性影响：`qat_kd=False` 时仍走原有 loss 接口；`teacher_preds` 仅在 detect KD 路径下传入
- 额外修复：resume 时补保留 `qat_kd`、`qat_kd_lambda`、`qat_kd_temperature`，避免续跑后 `kd_loss` 被静默旁路为 0
- 风险与回滚：若 KD loss 接口侵入过深，优先保持 detect 路径最小改动，必要时通过配置开关完全旁路
