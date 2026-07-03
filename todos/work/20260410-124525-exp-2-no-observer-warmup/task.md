# [Exp-2] 无预热全 INT8 长训练验证

**Status:** AwaitingCommit
**Agent PID:** 1097453
**Created At:** 2026-04-10 12:45:25

## Original Todo

Exp-2：恢复全 INT8 配置，验证 fake_quant 从 epoch 0 开启且延长训练是否能够提升 QAT 精度。超参：`lr0=5e-5`、`lrf=0.01`、`cos_lr=True`、`warmup_epochs=3`、`fliplr=0.5`、`epochs=100`。

## Description

### 目标

在不沿用 Exp-1 检测头 FP32 配置的前提下，启动一轮全 INT8 的长训练，验证训练策略本身是否还能继续逼近目标精度。

### 范围

- 包含：独立全 INT8 量化配置、Exp-2 启动命令、训练目录和初始状态记录
- 不包含：新的量化算子改动、KD-QAT 实现、分割实验

### 验收标准

- [x] Exp-2 使用独立全 INT8 quant config，不污染 Exp-1 配置
- [x] 训练以 `lr0=5e-5/lrf=0.01/cos_lr=True/warmup_epochs=3/fliplr=0.5/epochs=100` 启动
- [x] 新 run 成功进入 epoch 训练

_Read [analysis.md](./analysis.md) in full for detailed research and context_

## Implementation Plan

- [x] 新增 `config.exp2.int8.json`
- [x] 记录 Exp-2 假设、启动参数和观察点
- [x] 启动训练并确认 run 正常进入 epoch

## Validation

- Lint: `/home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -m py_compile ultralytics/engine/model.py` ✅
- Test: 启动参数核对 ✅ `qat_config=config.exp2.int8.json, epochs=100, lr0=5e-5, lrf=0.01, cos_lr=True, warmup_epochs=3, fliplr=0.5`
- Build/Run: `ULTRALYTICS_SKIP_DATASET_HASH=1 ... python - <<'PY' ... model.train(...)` ✅ 已完成到 `runs/detect/exp2-int8` 的 `epoch 23`；最佳 `metrics/mAP50-95(B)=0.37617`（epoch 7），最后 `0.37087`（epoch 23），当前 run 已停止

## Notes

- 设计决策：Exp-2 明确恢复全 INT8，不与 Exp-1 的 head FP32 配置复用
- 兼容性影响：不改现有 `config.json`，后续可并行保留 Exp-1 与 Exp-2
- 风险与回滚：若训练不稳定或收益有限，按 `quant_plan` 转向 Exp-4
- 当前运行：`runs/detect/exp2-int8`
- 阶段结论：到 `epoch 23` 为止，Exp-2 最佳仅 `37.617`，明显低于 Exp-1 的 `38.321`，也低于原先为 `epoch 30` 设定的接近 `38.3` 观察目标；仅靠训练日程调整的收益目前偏弱
