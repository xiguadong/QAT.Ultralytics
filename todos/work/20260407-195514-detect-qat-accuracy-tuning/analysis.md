# Analysis - yolo26n-qat 检测模型精度调优

## 1. 需求拆解

### 背景

仓库在 2026-03-17 完成了 QAT 训练链路打通（见 `todos/work/20260317-124152-start-detect-qat-train/`），已成功跑通 10 epoch 的 COCO 全集 QAT 训练。当前最优结果为 **mAP50-95 = 37.44**（epoch 7），目标精度为 **39.8**，差距约 2.36 mAP。

### 输入/输出

- 输入：`runs/detect/qat/results.csv`、`runs/detect/qat/args.yaml`、`weights/yolo26n.pt`、`config.json`
- 输出：修改后的训练脚本、trainer、文档，以及可复现的精度记录

---

## 2. 基准数据（已确认）

### 2.1 Float 基准

来源：`yolo val model=weights/yolo26n.pt data=coco.yaml`（2026-04-08 运行，faster-coco-eval 官方评估）

| 指标         | 值          |
| ------------ | ----------- |
| **mAP50-95** | **0.39949** |
| mAP50        | 0.55347     |
| mAP75        | 0.43459     |
| mAP-small    | 0.197       |
| mAP-medium   | 0.440       |
| mAP-large    | 0.581       |

### 2.2 QAT 实验汇总

| run          | 配置                            | QAT 最佳 mAP50-95 | 量化损耗  |
| ------------ | ------------------------------- | ----------------- | --------- |
| qat（基准）  | lr0=1e-5, 10ep, 无预热          | 37.44（ep7）      | -2.51     |
| qat2（本次） | lr0=5e-5, 50ep, observer预热5ep | 37.534（ep43）    | **-2.42** |

### 2.3 目标与现实

| 项目               | 数值         |
| ------------------ | ------------ |
| Float baseline     | 39.95        |
| **目标 QAT mAP**   | **39.8**     |
| 允许最大量化损耗   | **0.15 mAP** |
| 当前实际量化损耗   | **2.42 mAP** |
| **仍需弥合的差距** | **2.27 mAP** |

---

## 3. 根本原因分析（修订版）

### 3.1 qat2 两阶段现象

```
epoch 1-5  (fake_quant OFF, observer 校准)：39.30 → 39.65  ← 接近 float
epoch 6    (fake_quant ON, 首个 QAT epoch)：36.94           ← 骤降 2.66！
epoch 7-50 (fake_quant ON, QAT 训练)      ：37.09 → 37.53  ← 缓慢恢复，在 37.5 平台期停滞
```

### 3.2 结论

**问题不在 LR 或 epoch 数，而在量化噪声本身过大。**

量化损耗 ~2.4 mAP 是系统性问题，原因在于：

1. **observer 预热反效果**：5 个 epoch 的纯 float 训练使权重离"对量化友好的区域"更远。fake_quant 开启瞬间的冲击（-2.66 mAP）比直接从 epoch 0 开 QAT 更大，且模型需要更多 epoch 恢复。

2. **量化噪声来源**：
   - 激活量化：per-tensor U8（全局一个 scale），对分布差异较大的层误差大
   - 权重量化：per-channel S8（相对 OK）
   - 检测头对量化敏感：DFL 分布、box 回归的精细数值对量化误差容忍度低

3. **平台效应**：QAT phase 持续 44 个 epoch，mAP 始终在 37.0-37.5 震荡，从未突破。说明这是量化方案的精度上限，而非训练未收敛。

---

## 4. 修订方案（三步走）

### 方案 A：去掉 observer 预热，从 epoch 0 开启 fake_quant（优先验证）

**依据**：qat2 数据表明预热适得其反，fake_quant 应从训练开始即介入，让模型自始就学习对量化的鲁棒性。

**超参**：

| 参数                  | 值   | 理由                 |
| --------------------- | ---- | -------------------- |
| `qat_observer_epochs` | 0    | 不预热               |
| `lr0`                 | 5e-5 | 同 qat2              |
| `epochs`              | 100  | 更多时间适应量化噪声 |
| `cos_lr`              | True | 平滑衰减             |
| `warmup_epochs`       | 3    | 稳定早期训练         |
| `fliplr`              | 0.5  | 轻量正则             |

**预期**：验证去掉预热后能否超越 qat2 的 37.53，若能到 38+，方向正确。

---

### 方案 B：跳过敏感层量化（修改 config.json）

**依据**：检测头（Detect layer）对量化最敏感，可以通过 `regional_configs` 跳过或降低其量化激进度。

**修改 `config.json`**：

```json
{
  "global_config": {
    "is_symmetric": false,
    "input": { "dtype": "U8", "qmin": 0, "qmax": 255 },
    "weight": { "dtype": "S8", "qmin": -127, "qmax": 127 }
  },
  "regional_configs": [
    {
      "module_name_regex": ".*model\\.23.*",
      "config": null
    }
  ]
}
```

`"config": null` 表示跳过该层的量化。若目标硬件允许，可先跳过 head 验证精度上限。

**预期**：跳过 head 后 QAT mAP 应接近 float（39+），可确认损耗来源。

---

### 方案 C：知识蒸馏辅助 QAT（KD-QAT，需代码改动）

**依据**：KD-QAT 是当前业界达到高精度 INT8 量化的主流手段。float 模型作为 teacher，QAT 模型作为 student，用 KL 散度等软标签损失弥补量化差距。

**实现方式**：

- 在 `trainer.py` 的前向过程中，同时运行 `float_model(batch)` 和 `qat_model(batch)`
- 增加蒸馏损失项：`L = L_task + λ * L_kd`
- `L_kd`：对检测头的预测 logits 做 KL 散度

**工作量**：中等（约 100 行改动），需修改 trainer 前向和损失计算。

**预期**：业界经验 KD-QAT 可将量化损耗从 2+ 压缩到 0.5 以内，有望达到 39.5+。

---

## 5. 执行优先级

```
方案 B（30 分钟）→ 确认损耗是否主要来自 head
  ↓ 若 head 跳过后 QAT ≥ 39.5
方案 A（12 小时）→ 恢复 head 量化，验证更长训练能否恢复
  ↓ 若 A 仍不达标
方案 C（1-2 天）→ 实现 KD-QAT
```

---

## 6. 代码变更明细

### 已实施（qat2 实验）

| 文件                            | 变更                                                                       |
| ------------------------------- | -------------------------------------------------------------------------- |
| `train.py`                      | epochs 10→50, lr0 1e-5→5e-5, cos_lr, warmup, fliplr, qat_observer_epochs=5 |
| `ultralytics/engine/trainer.py` | 新增 `_apply_qat_observer_schedule()` 方法                                 |
| `ultralytics/cfg/default.yaml`  | 注册 `qat_observer_epochs: 0`                                              |

### 待实施

| 方案 | 文件                            | 变更描述                          |
| ---- | ------------------------------- | --------------------------------- |
| A    | `train.py`                      | qat_observer_epochs=0, epochs=100 |
| B    | `config.json`                   | 增加 regional_configs 跳过 head   |
| C    | `ultralytics/engine/trainer.py` | 增加 KD 损失分支                  |

---

## 7. 训练结果记录

### qat（基准，2026-03-17）

| epoch | mAP50-95    | lr     | 备注     |
| ----- | ----------- | ------ | -------- |
| 1     | 0.36038     | 1e-5   | —        |
| 7     | **0.37442** | 4.6e-6 | **最佳** |
| 10    | 0.37283     | 1.9e-6 | 结束     |

配置：lr0=1e-5, lrf=0.1, 10ep, 无 augment, 无 observer 预热

---

### qat2（2026-04-07～08）

| epoch  | mAP50-95    | lr     | 阶段             | 备注                       |
| ------ | ----------- | ------ | ---------------- | -------------------------- |
| 1      | 0.39298     | 2.5e-5 | observer 校准    | fake_quant 关              |
| 2      | 0.39506     | 5.0e-5 | observer 校准    | —                          |
| 3      | 0.39566     | 5.0e-5 | observer 校准    | —                          |
| **4**  | **0.39653** | 5.0e-5 | observer 校准    | **全局最佳（float 模式）** |
| 5      | 0.39602     | 4.9e-5 | observer 校准    | —                          |
| 6      | 0.36942     | 4.9e-5 | **QAT 首 epoch** | **骤降 2.66！**            |
| 9      | 0.37355     | 4.7e-5 | QAT              | —                          |
| 15     | 0.37408     | 4.1e-5 | QAT              | —                          |
| **43** | **0.37534** | 3.6e-6 | QAT              | **QAT 阶段最佳**           |
| 50     | 0.37194     | 5.5e-7 | QAT              | 结束                       |

配置：lr0=5e-5, lrf=0.01, cos_lr, 50ep, fliplr=0.5, qat_observer_epochs=5

**关键结论**：observer 预热 5 epoch 无益，QAT 阶段量化损耗约 2.42 mAP，在 37.0-37.5 平台期停滞，与 qat 基准相比无本质改善。

---

## 8. Float 基准

来源：`yolo val model=weights/yolo26n.pt data=coco.yaml device=2 batch=64`（2026-04-08）

```
mAP50-95 = 0.39949
mAP50    = 0.55347
mAP75    = 0.43459
```

**结论：目标 39.8 允许量化损耗 ≤ 0.15 mAP。当前损耗约 2.42 mAP，需从根本上改变量化策略。**

---

## 9. 风险与回滚

| 风险                          | 对策                                                   |
| ----------------------------- | ------------------------------------------------------ |
| 方案 B 跳过 head 后硬件不支持 | 先确认目标硬件是否允许混合量化，若不允许则只做方案 A/C |
| 方案 A（无预热）QAT 不稳定    | warmup_epochs=3 缓解；若 loss NaN 则降 lr0 到 2e-5     |
| 方案 C KD 损失权重难调        | 先用 λ=1.0 跑 5 epoch 看趋势                           |
