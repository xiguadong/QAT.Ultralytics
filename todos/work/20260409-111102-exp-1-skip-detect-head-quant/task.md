# [Exp-1] 跳过检测头量化敏感性分析

**Status:** AwaitingCommit
**Agent PID:** 1097453
**Created At:** 2026-04-09 11:11:20

## Original Todo

Exp-1：敏感性分析，跳过检测头量化。修改 `config.json`，在 `regional_configs` 中为 `model.23` 下的所有 conv 节点指定 null 量化（FP32 直通）；同步修改 `ultralytics/utils/ax_quantizer_utils.py` 的 conv annotator 和 `ultralytics/utils/ax_quantizer.py` 的 `load_regional_config`，并完成 10 epoch QAT 验证与结论归档。

## Description

### 目标

让 PT2E QAT quantizer 支持通过 regional config 为指定 conv 节点配置 `module_config: null`，从而让 YOLO26 检测头 `model.23.*` 在 QAT 图中保持 FP32。

### 范围

- 包含：`ax_quantizer.py` 的 regional 配置加载、`ax_quantizer_utils.py` 的 conv regional override、`config.json` 的 head 节点配置、最小图验证
- 不包含：KD-QAT、新训练策略、硬件混合精度部署确认

### 验收标准

- [x] `module_config: null` 不再导致 quantizer 配置加载报错
- [x] `config.json` 可指定检测头 conv 节点跳过量化
- [x] 最小验证可确认 head 节点不再插入 QAT fake quant

_Read [analysis.md](./analysis.md) in full for detailed research and context_

## Implementation Plan

- [x] 修改 `ultralytics/utils/ax_quantizer.py`，支持 regional `module_config=None`
- [x] 修改 `ultralytics/utils/ax_quantizer_utils.py`，让 conv regional override 可清除已有量化 qspec
- [x] 更新 `config.json`，为 `model.23.*` 对应的 conv 节点配置 FP32 直通
- [x] 运行最小验证并记录结果
- [x] 启动 Exp-1 训练并确认进入 QAT epoch

## Validation

- Lint: `/home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -m py_compile ultralytics/utils/ax_quantizer.py ultralytics/utils/ax_quantizer_utils.py ultralytics/utils/qat_utils.py` ✅
- Test: 最小图验证 ✅ `annotated_head_count=48, annotated_head_all_none=True, prepared_head_count=12, prepared_head_all_none=True`
- Build/Run: `ULTRALYTICS_SKIP_DATASET_HASH=1 /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python train.py` ✅ 已完成 `runs/detect/qat3` 10 epoch；最佳 `metrics/mAP50-95(B)=0.38321`（epoch 9），最终 `0.38180`（epoch 10）

## Notes

- 设计决策：沿用现有 `module_names` 精确匹配 FX conv 节点名，不引入新的 regex 配置语义
- 兼容性影响：仅 regional `module_config=null` 的 conv override 会改变行为，其他配置路径保持不变
- 风险与回滚：为兼容 PT2E `prepare_qat_pt2e()`，除了清空 conv 自身 qspec，还同步清除了下游 `view_*` 上残留的 `SharedQuantizationSpec` 引用；若后续发现误伤，可回退到更窄的用户链清理逻辑
- 实验结论：跳过检测头量化能把 QAT 从 `37.53` 提升到 `38.321`，但仍低于 `38.5` 决策阈值，也明显低于目标 `39.2`；说明 head 不是唯一主因，backbone/neck 或训练策略仍是主要瓶颈
- 后续建议：若严格按决策树推进，优先级应转向 `Exp-4`；若仍想继续验证训练策略，可保留 `Exp-2` 作为次优先级实验，但应恢复全 INT8 配置后再跑
