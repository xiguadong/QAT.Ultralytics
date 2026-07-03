# yolo26n-qat 检测模型精度调优

**Status:** InProgress
**Created At:** 2026-04-07 19:55:14
**Updated At:** 2026-04-08

## Original Todo

阅读 runs/detect/qat 作为 QAT 基准，实现 yolo26n 模型精度达到 39.8+ mAP50-95

## Description

### 目标

分析现有 QAT 基准训练结果，定位精度瓶颈，通过调参与代码修改使 yolo26n-qat 在 COCO val 上达到 mAP50-95 ≥ 39.8。

### 范围

- 包含：基准分析、超参调整、QAT 训练策略、量化配置、KD-QAT（必要时）
- 不包含：模型架构修改、分割模型

### 验收标准

- [ ] COCO val mAP50-95 ≥ 39.8（fake_quant 完全开启状态下）
- [x] Float 基准已确认：**39.95 mAP50-95**
- [x] 修改有文档记录，可追溯可复现

## 当前状态（2026-04-08）

已完成两轮实验，确认核心问题：**量化损耗约 2.4 mAP，远超允许的 0.15 mAP 上限。**

| run | QAT 最佳 mAP | 量化损耗 |
|-----|------------|---------|
| qat（基准） | 37.44 | -2.51 |
| qat2（50ep + observer 预热） | 37.53 | -2.42 |
| **目标** | **39.80** | **-0.15** |

## 下一步计划

- [ ] 方案 B：修改 `config.json`，跳过检测头量化，定位损耗来源（30 分钟）
- [ ] 方案 A：去掉 observer 预热，从 epoch 0 开启 fake_quant，更长训练（12 小时）
- [ ] 方案 C：实现 KD-QAT 蒸馏损失（若 A/B 仍不达标）

## Implementation Plan（已完成）

- [x] 读取 runs/detect/qat，分析基准
- [x] 定位精度瓶颈（LR 过小 + epoch 不足 + 无 observer 预热）
- [x] 实施 qat2：lr0=5e-5, 50ep, cos_lr, observer 预热 5ep
- [x] 新增 `_apply_qat_observer_schedule()` 方法
- [x] 注册 `qat_observer_epochs` 参数
- [x] 确认 float 基准（39.95）
- [x] 重规划文档，明确新方向

## Validation

- Float val：`mAP50-95=0.39949`（2026-04-08, faster-coco-eval 官方评估）
- qat2 run：50 epoch 全部完成，QAT 阶段最佳 37.534（epoch 43）
- 核心结论：问题在量化噪声本身，需改变量化策略（见 analysis.md §4）

_Read [analysis.md](./analysis.md) in full for detailed research and context_
