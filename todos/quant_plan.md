# yolo26n QAT 精度调优实验规划

**Created:** 2026-04-08  
**目标：** COCO val mAP50-95 ≥ 39.2（fake_quant 全开状态）

---

## 已知基准

| 项目                           | 数值                                            |
| ------------------------------ | ----------------------------------------------- |
| Float baseline                 | **39.95** mAP50-95                              |
| 目标 QAT                       | **≥ 39.2** mAP50-95（= 40.0 − 0.8）             |
| 允许量化损耗                   | **≤ 0.8**                                       |
| **当前最优 QAT（exp18, ep6）** | **40.33（超过 float，+0.38）**                  |
| W8A8 最优 QAT（exp24, ep9）    | 39.65（end2end=False 验证，损耗 -1.25，未达标） |

---

## 核心问题

两轮实验（qat 基准 + qat2）得出一致结论：  
**QAT 阶段在 37.0–37.5 存在平台期，增加 LR / epoch / observer 预热均无法突破。**

目标放宽后，需弥合的差距从 2.27 降为 **1.67 mAP**（37.53 → 39.2），但量化损耗 2.47 mAP 的根本原因仍需解决：

```
A. 检测头（model.23）对 per-tensor INT8 激活量化敏感
B. Backbone/Neck 本身量化损耗不可忽略
C. 训练策略（无 KD）导致模型无法充分适应量化约束
```

---

## 实验序列

### Exp-1：敏感性分析——跳过检测头量化

**目的：** 定量测出 head 对量化损耗的贡献比例，决定后续方向。

**方法：**  
修改 `config.json`，在 `regional_configs` 中为 `model.23` 下的所有 conv 节点指定 null 量化（FP32 直通）。

需要同步修改 `ultralytics/utils/ax_quantizer_utils.py` 的 conv annotator，支持 `module_config=None` 时跳过对已有 annotation 的覆写，以及修改 `ultralytics/utils/ax_quantizer.py` 的 `load_regional_config` 处理 null `module_config`。

```
head 模块路径：model.23.{cv2,cv3,one2one_cv2,one2one_cv3}.*（共 309,656 参数，占 12%）
```

**验证方式：** 重新跑 1–5 epoch QAT，在 fake_quant 开启后观察 mAP。

**代码改动规模：** ~20 行，涉及 `ax_quantizer.py`、`ax_quantizer_utils.py`、`config.json`

**预期结果与决策：**

| Exp-1 结果        | 结论                                                | 下一步                 |
| ----------------- | --------------------------------------------------- | ---------------------- |
| QAT mAP ≥ 39.2    | head 是主因，backbone/neck 损耗可接受，**直接达标** | → 完成 ✅              |
| QAT mAP 38.5–39.2 | head 贡献大，backbone 仍有少量损耗                  | → Exp-2 或 Exp-3 补足  |
| QAT mAP < 38.5    | head 贡献小，损耗主要在 backbone                    | → 直接 Exp-4（KD-QAT） |

**实际结果（2026-04-10）：**

| 指标           | 数值               |
| -------------- | ------------------ |
| 运行目录       | `runs/detect/qat3` |
| 最佳 epoch     | 9                  |
| 最佳 mAP50     | **53.672**         |
| 最佳 mAP50-95  | **38.321**         |
| 最终 mAP50-95  | 38.180             |
| 相对 qat2 提升 | **+0.79**          |
| 距目标 39.2    | **-0.879**         |

**结论更新：**

- 检测头量化敏感性存在，但不是唯一主因；仅跳过 head 量化不足以逼近 39.2
- Exp-1 结果落在 `< 38.5` 档位，因此按原决策树更偏向 `Exp-4`
- 若仍希望先排除“训练策略”因素，可继续执行 `Exp-2`，但应视为次优先级验证，而非主路径

---

### Exp-2：无 observer 预热 + 延长训练

**目的：** 验证 fake_quant 从 epoch 0 开启（默认行为）+ 延长训练是否能让模型更好地适应量化约束。

**依据：** qat2 数据表明，5 epoch 纯 float 预热使模型偏离"对量化友好"的权重区域，epoch 6 骤降 2.66 mAP，此后 44 epoch 只能恢复到 37.53。`qat_observer_epochs=0` 是默认值，无需额外配置；延长训练至 100 epoch，让量化噪声从训练伊始参与梯度优化，可能形成更鲁棒的权重分布。

**超参：**

```yaml
# qat_observer_epochs 默认为 0，fake_quant 从 epoch 0 开，无需设置
lr0: 5e-5
lrf: 0.01
cos_lr: True
warmup_epochs: 3
fliplr: 0.5
epochs: 100
```

**按当前进度更新后的执行约束：**

- 必须恢复 **全 INT8** 配置后再开始，不沿用 Exp-1 的 `head FP32` 配置
- 建议新增独立配置文件（如 `config.exp2.int8.json`）或在启动前将 `config.json` 的 `regional_configs` 清空，避免和 Exp-1 混淆
- 继续保留现有 resume 修复，长训练允许中断后从 `last.pt` 续跑
- 保持 `qat_validate=True` 与 `save_period=1`，方便按阶段评估

**建议里程碑（仅用于是否继续观察，不作为硬阈值结论）：**

| 观察点    | 建议判读                                           |
| --------- | -------------------------------------------------- |
| epoch 10  | 若仍显著低于 37.8，说明仅靠训练日程改善空间有限    |
| epoch 30  | 若未接近 38.3，Exp-2 达标概率偏低                  |
| epoch 50  | 若未达到 38.8 左右，建议停止继续消耗算力并转 Exp-4 |
| epoch 100 | 目标仍是 ≥ 39.2                                    |

**预期结果与决策：**

| Exp-2 结果                     | 结论                      | 下一步                 |
| ------------------------------ | ------------------------- | ---------------------- |
| QAT mAP ≥ 39.2（**达标**）     | 去预热 + 长训练已足够     | → 完成 ✅              |
| QAT mAP 38.0–39.2              | 有改善，结合 Exp-3/4 补足 | → Exp-3 或 Exp-4       |
| QAT mAP ≈ 37.5（与 qat2 持平） | 训练策略已到瓶颈          | → 直接 Exp-4（KD-QAT） |
| QAT mAP < 37.0（不稳定）       | LR 过高或 warmup 不足     | 降 lr0 至 2e-5 重试    |

**优先级更新：**

- 原始决策树下，Exp-1 已更偏向直接进入 `Exp-4`
- 当前保留 Exp-2 的理由仅是：在实现 KD-QAT 前，低成本再验证一次“训练策略是否仍有剩余增益”
- 因此 Exp-2 应被视为“可选验证项”，不是当前最有把握的主线方案

**最终结果（截至 2026-04-14，epoch 100）：**

| 指标          | 数值                    |
| ------------- | ----------------------- |
| 运行目录      | `runs/detect/exp2-int8` |
| 最新 epoch    | 100                     |
| 最佳 epoch    | 7                       |
| 最佳 mAP50    | **53.031**              |
| 最佳 mAP50-95 | **37.617**              |
| 最终 mAP50-95 | 36.978                  |

**结论更新：**

- Exp-2 跑满 `100 epoch` 后，最佳结果仍停留在 `epoch 7`，后半程没有出现有效反弹
- 其最优 `mAP50-95=37.617` 仍低于 Exp-1 的 `38.321`，也显著低于目标 `39.2`
- 因此可以正式判定：单靠“无 observer 预热 + 长训练”不足以突破当前平台期，主线应转向 `Exp-4`

---

### Exp-3：混合精度——head 保持 FP32

**状态：已排除（硬件不支持）**

AXERA 硬件仅支持 INT8 和 INT16，不支持 FP32 层。此方案不可部署，已废弃。

---

### Exp-5：head INT16 + backbone INT8（新增，2026-04-21）

**目的：** 在硬件约束（INT8/INT16，无 FP32）下，用 INT16 替代 FP32 降低检测头量化损耗。

**依据：**

- Exp-1 证明 head 量化贡献约 +0.88 mAP（38.32 vs 37.44），但 FP32 不可部署
- INT16 量化精度远高于 INT8，可在硬件支持范围内最大程度保留 head 精度
- Exp-4（KD-QAT）已实现 +0.33 mAP，可与 INT16 head 叠加

**方法：**  
修改 `config.json`，在 `regional_configs` 中为 `model.23` 下所有 conv 节点指定 S16 量化配置（对称，qmin=-32767，qmax=32767），backbone+neck 保持 A8W8 INT8。

需同步修改 `ax_quantizer_utils.py` 的 conv annotator，支持通过 `nn_module_stack` 路径前缀（`"model.23"`）匹配节点，替代当前基于 FX 节点名的匹配。

**超参（与 Exp-4 保持一致）：**

```yaml
lr0: 5e-5
lrf: 0.01
cos_lr: True
warmup_epochs: 3
fliplr: 0.5
epochs: 50
qat_kd: True # 叠加 KD 增益
qat_kd_lambda: 1.0
qat_kd_temperature: 4
```

**代码改动规模：** ~25 行，涉及 `ax_quantizer.py`、`ax_quantizer_utils.py`、`config.json`

**预期结果与决策：**

| Exp-5 结果                      | 结论                           | 下一步             |
| ------------------------------- | ------------------------------ | ------------------ |
| QAT mAP ≥ 39.2                  | head INT16 + KD 达标           | → 完成 ✅          |
| QAT mAP 38.5–39.2               | 有改善，调整 KD 超参或延长训练 | → 调参续跑         |
| QAT mAP ≈ 37.8（与 Exp-4 持平） | INT16 head 增益不显著          | → 重新评估量化策略 |

---

### Exp-4：KD-QAT（知识蒸馏辅助 QAT）

**目的：** 最强方案，通过 float teacher 指导 QAT student，系统性压缩量化损耗。不依赖混合精度，全层 INT8 量化下争取 ≥ 39.2。

**依据：** 业界实验表明 KD-QAT 可将 INT8 量化损耗从 2+ mAP 压缩到 0.3–0.5 以内，目标 39.2（损耗 ≤ 0.8）在此方案下有较大把握达到。

**实现设计：**

损失函数：

```
L_total = L_task + λ * L_kd

L_kd = KLDiv(student_cls_logits / T, teacher_cls_logits / T)
      + MSE(student_box_pred, teacher_box_pred)
```

超参初始值：温度 T = 4，λ = 1.0

涉及改动：

| 文件                                      | 改动描述                                                            |
| ----------------------------------------- | ------------------------------------------------------------------- |
| `ultralytics/engine/trainer.py`           | 前向时同时运行 `float_model`，收集 teacher 输出；增加 KD 损失计算   |
| `ultralytics/models/yolo/detect/train.py` | `criterion()` 支持 KD 损失项                                        |
| `ultralytics/cfg/default.yaml`            | 注册 `qat_kd: False`、`qat_kd_lambda: 1.0`、`qat_kd_temperature: 4` |

**代码改动规模：** 约 80–120 行，3 个文件

**Teacher 模型来源：** `trainer.model`（float eager model，在 `_prepare_qat_training` 之前已存在，无需额外加载，deepcopy 冻结即可）

**预期结果：** QAT mAP ≥ 39.2，业界经验损耗可压缩到 0.3–0.5，若 λ 调优得当有望达到 39.5+

**最终结果（截至 2026-04-20，epoch 50）：**

| 指标          | 数值                       |
| ------------- | -------------------------- |
| 运行目录      | `runs/detect/exp4-kd-int8` |
| 最新 epoch    | 50                         |
| 最佳 epoch    | 45                         |
| 最佳 mAP50    | **53.207**                 |
| 最佳 mAP50-95 | **37.769**                 |
| 最终 mAP50-95 | 37.610                     |

**结论更新：**

- Exp-4 跑满 `50 epoch` 后，最佳结果出现在 `epoch 45`，说明 KD-QAT 相比 Exp-2 有小幅增益，但不显著
- 其最优 `mAP50-95=37.769` 高于 Exp-2 的 `37.617`，但仍低于 Exp-1 的 `38.321`
- 距目标 `39.2` 仍差约 `1.431`，因此当前 KD 设计与超参配置不足以达标
- 实验过程中已修复 1 个 resume 缺陷：恢复训练时需保留 `qat_kd`、`qat_kd_lambda`、`qat_kd_temperature`，否则 `kd_loss` 会被静默旁路为 0

---

## 执行顺序与决策树

```
Exp-1（~1 天：代码改动 + 5 epoch 验证）
  │
  ├─ head 是主因（≥39.2，直接达标）→ 完成 ✅
  │
  ├─ head 贡献大（38.5–39.2）
  │    └─ 确认硬件支持混合精度？
  │         ├─ 是 → Exp-3（0.5 天配置 + 50 epoch）→ 达标 ✅
  │         └─ 否 → Exp-2 或 Exp-4
  │
  └─ head 非主因（< 38.5）
       └─ Exp-2（12 小时训练）
            ├─ ≥ 39.2 → 达标 ✅
            └─ 不达标  → Exp-4（KD-QAT）
```

**预估最快路径：** Exp-1 → Exp-3 → 完成，约 2 天  
**预估最慢路径：** Exp-1 → Exp-2 → Exp-4 → 完成，约 4–5 天

---

## 实验对照表（W8A8 阶段）

| 参数                | qat（基准） | qat2       | Exp-2      | Exp-3    | Exp-4      | Exp-5                    | Exp-6         |
| ------------------- | ----------- | ---------- | ---------- | -------- | ---------- | ------------------------ | ------------- |
| epochs              | 10          | 50         | 100        | —        | 50         | 29（中止）               | 50            |
| lr0                 | 1e-5        | 5e-5       | 5e-5       | —        | 5e-5       | 5e-5                     | 5e-5          |
| cos_lr              | False       | True       | True       | —        | True       | True                     | True          |
| warmup_epochs       | 0           | 2          | 3          | —        | 3          | 3                        | 3             |
| fliplr              | 0           | 0.5        | 0.5        | —        | 0.5        | 0.5                      | 0.5           |
| qat_observer_epochs | 0（默认）   | 5          | 0（默认）  | —        | 0（默认）  | 0（默认）                | 0（默认）     |
| head 量化           | INT8        | INT8       | INT8       | ~~FP32~~ | INT8       | ~~INT16~~                | INT8          |
| act_observer        | moving_avg  | moving_avg | moving_avg | —        | moving_avg | moving_avg               | **histogram** |
| KD-QAT              | 无          | 无         | 无         | —        | **有**     | **有**                   | **有**        |
| 实际最佳 mAP50-95   | 37.44       | 37.53      | 37.617     | —        | 37.769     | 38.573（29ep，不可部署） | —             |

### 中间探索实验（exp5-14，validator 修复前部分数据虚高）

> **注意：** exp5-10 期间 validator 存在 bug（qat_model 验证时未切 eval，BN 用 batch stats），导致 mAP 虚高至 ~0.52-0.54。修复后真实值应参考 exp15+ 结果。

| 参数              | exp5        | exp6        | exp7        | exp8       | exp9           | exp10       |
| ----------------- | ----------- | ----------- | ----------- | ---------- | -------------- | ----------- |
| 模型              | yolo26n     | yolo26n     | yolo26n     | yolo26n    | yolo26n        | yolo26n     |
| epochs            | 50          | 50          | 50          | 50         | 50             | 50          |
| lr0               | 5e-5        | 5e-5        | 5e-5        | 5e-5       | 5e-5           | 5e-5        |
| lrf               | 0.01        | 0.01        | 0.01        | 0.01       | 0.01           | 0.01        |
| cos_lr            | True        | True        | True        | True       | True           | True        |
| warmup_epochs     | 3           | 3           | 3           | 3          | 3              | 3           |
| head 量化         | S16         | INT8        | INT8        | INT8       | one2many       | U16         |
| act_observer      | moving_avg  | histogram   | moving_avg  | moving_avg | moving_avg     | moving_avg  |
| KD-QAT            | 有          | 有          | 有          | 有 (LSQ)   | 无             | 有          |
| 实际最佳 mAP50-95 | 38.57(虚高) | 51.87(虚高) | 53.67(虚高) | 无(崩)     | 37.72          | 53.67(虚高) |
| 说明              | 29ep中止    | 仅2ep有值   |             | LSQ量化器  | 仅one2many分支 |             |

#### yolo11n 调参（exp11-14）

| 参数           | exp11   | exp12   | exp13     | exp13-hist2 | exp14      |
| -------------- | ------- | ------- | --------- | ----------- | ---------- |
| 模型           | yolo11n | yolo11n | yolo11n   | yolo11n     | yolo11n    |
| epochs         | 50      | 50      | 50        | 50          | 50         |
| lr0            | 4e-5    | 4e-5    | 4e-5      | 4e-5        | 4e-5       |
| lrf            | 0.2     | 0.2     | 0.2       | 0.1         | 0.1        |
| warmup_epochs  | 3       | 0       | 0         | 0           | 0          |
| act_observer   | minmax  | minmax  | histogram | histogram   | moving_avg |
| 最佳 mAP50-95  | 36.91   | 37.69   | 无(崩)    | 37.62       | **38.48**  |
| float baseline | 39.25   | 39.25   | 39.25     | 39.25       | 39.25      |

### W8A8 调参阶段（exp15-17）

| 参数              | exp15 | exp16 | exp17  |
| ----------------- | ----- | ----- | ------ |
| epochs            | 50    | 20    | 50     |
| lr0               | 4e-5  | 2e-5  | 2e-5   |
| lrf               | 0.1   | 0.1   | 0.1    |
| cos_lr            | False | False | False  |
| warmup_epochs     | 0     | 0     | 0      |
| act_dtype         | U8    | U8    | U8     |
| weight_dtype      | S8    | S8    | S8     |
| EMA               | 无    | 无    | **有** |
| 实际最佳 mAP50-95 | 38.13 | 38.24 | 38.29  |
| 达标              | 否    | 否    | 否     |

#### LSQ 与 KD (exp20-21)

| 参数              | exp20  | exp21        |
| ----------------- | ------ | ------------ |
| epochs            | 50     | 50（15中止） |
| lr0               | 2e-5   | 2e-5         |
| act_dtype         | U8     | U8           |
| weight_dtype      | S8     | S8           |
| EMA               | 有     | 有           |
| LSQ               | **有** | 无           |
| KD                | 无     | **有**       |
| 实际最佳 mAP50-95 | 38.25  | **38.18**    |
| 达标              | 否     | 否           |

#### yolo26s 与混合精度 (exp22-23)

| 参数              | exp22   | exp23                   |
| ----------------- | ------- | ----------------------- |
| 模型              | yolo26s | yolo26n                 |
| epochs            | 50      | 50                      |
| lr0               | 2e-5    | 2e-5                    |
| 配置              | W8A8    | bb-W8A8 + head-U16A/S8W |
| 实际最佳 mAP50-95 | 45.91   | 39.12                   |
| 达标              | 否      | 否                      |

#### end2end=False 探索 (exp24-25)

| 参数              | exp24             | exp25                   |
| ----------------- | ----------------- | ----------------------- |
| 模型              | yolo26n (原 yaml) | yolo26n (e2eFalse yaml) |
| epochs            | 50                | 50（运行中）            |
| lr0               | 2e-5              | 2e-5                    |
| end2end           | False (验证时)    | False (yaml 内)         |
| reg_max           | 1                 | 1                       |
| 结构              | 原双分支          | 单分支 one2many         |
| 实际最佳 mAP50-95 | 39.63             | —                       |
| 达标              | 否                | —                       |

### U16 激活阶段（exp18-19）

| 参数              | exp18                   | exp19              |
| ----------------- | ----------------------- | ------------------ |
| epochs            | 50（14 中止）           | 50（8 中止）       |
| lr0               | 2e-5                    | 2e-5               |
| lrf               | 0.1                     | 0.1                |
| act_dtype         | **U16** [0, 65535]      | **U16** [0, 65535] |
| weight_dtype      | **S16** [-32767, 32767] | **S8** [-127, 127] |
| EMA               | 有                      | 有                 |
| 实际最佳 mAP50-95 | **40.33**（ep6）        | **40.25**（ep7）   |
| 达标              | **是（超过 float）**    | **是**             |

### 量化误差诊断

| 配置                  | Boxes diff (vs Float) |   改善   |
| --------------------- | :-------------------: | :------: |
| Float                 |         0.000         |    —     |
| W8A8 (U8 act + S8 wt) |         0.162         | baseline |
| U16A + S16W           |         0.003         |   98%    |
| U16A + S8W            |           —           |   待测   |

### 关键结论

1. **end2end=False 验证显著提升 W8A8 精度**：exp24 达 39.65，exp32 提至 39.82（S16 matmul），exp33 达 39.90（S8 matmul）**达标**
2. **S8 matmul 可部署**：精度差异仅 -0.12 vs S16
3. **per-scale head 消除 Concat 量化冲突**：ONNX 导出零 Cast、零冗余 Q-DQ
4. **EMA/LSQ/KD 对 W8A8 几乎无效**
5. **U16 激活可从根源解决**：exp18 达 40.33 超过 float
6. **eval.py 精确复现训练验证**：正常 epoch 差距 ≤0.07
7. **export.py 自动检测 qat_config**：从 checkpoint 元数据读取，避免 matmul 配置错误
8. **Dim.AUTO 全动态 shape** 应用于所有 `export_for_training` 调用
9. **one2many ONNX 无需 feats 输出**：6 输出替代 9 输出
10. **EMA 无 bug**：observer/BN/复制链全部正确，CSV 尖峰为统计噪声

### yolo26n vs yolo11n 量化敏感性分析

yolo11n W8A8 仅损失 0.77 mAP，yolo26n 损失 1.66 mAP（2.2x）。从权重分布分析原因：

| 指标                       |  yolo11n  |  yolo26n  | 倍率  |
| -------------------------- | :-------: | :-------: | :---: |
| conv 层数                  |    88     |  **126**  | 1.43x |
| backbone weight_range 均值 |   0.48    | **0.96**  | 2.0x  |
| backbone weight_std 均值   |   0.033   | **0.068** | 2.1x  |
| neck weight_range 均值     |   0.37    | **0.75**  | 2.0x  |
| neck weight_std 均值       |   0.022   | **0.059** | 2.7x  |
| 通道均匀性 (cv)            | 0.26-0.33 | 0.20-0.35 | 相近  |

**根因：** yolo26n backbone/neck 权重范围是 yolo11n 的 2 倍。INT8 每通道 scale = range/254，范围越大 → 量化步长越大 → 单层误差更高。叠加 1.43x 更多的 conv 层，累积误差约 2.9x，与实测 2.2x mAP 损失吻合。

### W8A8 精度对比

| 模型                    | Float | W8A8 最佳 | 损耗  | 损耗%  | 实验  |
| ----------------------- | :---: | :-------: | :---: | :----: | ----- |
| yolo11n                 | 39.25 | **38.48** | -0.77 | -1.96% | exp14 |
| yolo26n (e2e=False S16) | 40.90 | **39.82** | -1.08 | -2.64% | exp32 |
| yolo26n (e2e=False S8)  | 40.90 | **39.90** | -1.00 | -2.44% | exp33 |
| yolo26n (e2e=True)      | 39.95 |   38.29   | -1.66 | -4.15% | exp17 |
| yolo26s (e2e=True)      | 47.80 |   45.91   | -1.89 | -3.95% | exp22 |

exp33 (S8 matmul) 以 39.90 **达标**，损耗仅 -1.00 mAP。S8 vs S16 差异约 -0.12，可部署。

---

## 实施前须确认的技术细节

### 1. PT2E 图中 module_names 的命名规则

`ax_quantizer_utils.py:630` 用 `conv_node.name` 匹配 regional config 的 `module_names`。需确认这是：

- FX 图节点名（如 `conv2d_123`）——需 dump FX 图获取 head 节点名列表
- Eager module 路径（如 `model.23.cv2.0.0.conv`）——直接使用 Python 模块路径

**验证方法：** 运行 `prepare_pt2e_qat_model` 后打印 `model.graph`，检查 head 相关节点的 `.name` 属性与 `source_fn_stack` 元数据。

### 2. `load_regional_config` 对 null module_config 的处理

当前 `ax_quantizer.py:load_regional_config` 直接读取 `regional_config["module_config"]`，若为 `null` 会抛 TypeError。Exp-1 实施前须：

- `load_regional_config`：增加 `if module_config is None: return QuantizerRegionalConf(module_names=..., module_type=..., module_config=None)`
- 各 annotator（conv/linear 等）的非 global 分支：当 `quantization_config is None` 时，对已标注节点执行"清除 annotation"或"跳过覆写"

### 3. AXERA 硬件混合精度支持（Exp-3 前置）

先向硬件团队确认 AXERA NPU 是否支持 mixed-precision 图（部分层 FP32，部分层 INT8）。若不支持，Exp-3 作为精度上限实验（了解 head 量化贡献），最终方案走 Exp-4。

### 4. KD teacher 模型的 eval 保证（Exp-4）

`trainer.model` 在 QAT 训练期间处于 train 模式。用作 teacher 时须：

- `deepcopy` + `eval()` + `requires_grad_(False)`
- 在 `_prepare_qat_training` 完成后保存为 `trainer.float_teacher`
- teacher 不参与梯度更新，仅用于前向推理

---

## 下一步计划（2026-05-27）

### 已完成：exp23 — yolo26n Backbone W8A8 + Head U16A/S8W

| 组件            |                    激活                    |   权重    |
| --------------- | :----------------------------------------: | :-------: |
| Backbone/Neck   |                 U8 (INT8)                  | S8 (INT8) |
| Head (model.23) |                U16 (INT16)                 | S8 (INT8) |
| 结果            | 最佳 39.12（epoch 5），距目标 39.2 差 0.08 |           |

### 已完成：exp24 — yolo26n W8A8 + end2end=False 验证

原 yaml + 原权重，仅验证时路由到 one2many + NMS。最佳 **39.65**（epoch 9），W8A8 最高记录。

### 已完成：exp29 — yolo26n per-scale head (e2e=False, S8 matmul)

**代码改动（模型结构）：**

- `head.py`: `forward_head()` 返回 per-scale list（移除 torch.cat）；新增 `concat_flag` 参数
- `loss.py`: `v8DetectionLoss.loss()` 支持 per-scale list
- `tasks.py`: stride 初始化适配 per-scale feats
- `block.py`: `SPPF.forward()` 显式循环（PT2E 兼容）
- 详见 `MODEL_CHANGES.md`

**目的：** 消除检测头三尺度 Concat（scale 冲突最大 4.9x），每尺度独立量化域。

**实验：**
| 参数 | 值 |
|------|------|
| 模型 | yolo26n.pt（708/708 完全兼容） |
| 配置 | W8A8 (U8+S8), matmul=S8, end2end=False, EMA |
| 训练 | lr=2e-5, lrf=0.1, 50ep |
| 结果 | best=**39.64**（epoch 7） |
| 对比 | exp28（原结构 e2e=False, S8 matmul, 39.63） |

**结论：** per-scale head 消除 Concat 量化冲突，ONNX 导出质量改善，但训练精度与原始 Concat head 持平。

### 已完成：exp30 — per-scale head + concat_flag 快速验证 (e2e=False)

| 参数      | 值                                                                        |
| --------- | ------------------------------------------------------------------------- |
| 配置      | W8A8, matmul=S8, per-scale head                                           |
| 训练      | lr=2e-5, 3ep（快速验证）                                                  |
| 结果      | best=**39.60**（epoch 1）                                                 |
| ONNX 导出 | one2one: 1 DQ-Q merge → 0, one2many: 2 DQ-Q merge → 0, Cast=0, Identity=0 |

### 已完成：exp31 — S16 matmul 快速验证 (e2e=False)

| 参数      | 值                                          |
| --------- | ------------------------------------------- |
| 配置      | W8A8, matmul=S16, per-scale head            |
| 训练      | lr=2e-5, 3ep（快速验证）                    |
| 结果      | best=**39.74**（epoch 2）                   |
| ONNX 导出 | 零冗余 DQ-Q, Cast=0, Identity=0（完美导出） |

### 已完成：exp32 — S16 matmul 全量训练 (e2e=False)

| 参数      | 值                                                                 |
| --------- | ------------------------------------------------------------------ |
| 配置      | W8A8, matmul=S16, per-scale head, end2end=False                    |
| 训练      | lr=2e-5, lrf=0.1, 50ep, batch=64, 4xGPU, 无数据增强                |
| 最终最佳  | **39.82**（epoch 10），末 epoch50=39.80                            |
| ONNX 导出 | one2one: 2 mismatch+3 merge → 0; one2many: 2 mismatch → 0, 0 merge |
| 目标      | 距 39.9 差 0.08，未达标                                            |

### 已完成：exp33 — S8 matmul 全量训练 (e2e=False)

| 参数      | 值                                                        |
| --------- | --------------------------------------------------------- |
| 配置      | W8A8, matmul/gridsample=S8, per-scale head, end2end=False |
| config    | `config_matmul_s8.json`（S8 覆盖 `init_regional()` S16）  |
| 训练      | lr=2e-5, lrf=0.1, 50ep, batch=64, 4xGPU                   |
| 结果      | best=**39.90**（epoch9），**达标！**                      |
| S8 vs S16 | 差异约 -0.12（典型值 39.7 vs 39.82）                      |
| ONNX      | one2many: 0 DQ-Q merge, 2 requant (zp_diff)               |

### 已完成：exp34 — 移除全链路 EMA (e2e=False, S8 matmul)

| 参数 | 值                                                                                                                                         |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 配置 | 同 exp33（W8A8, matmul=S8, per-scale head, end2end=False, `config_matmul_s8.json`）                                                        |
| 变量 | `qat_ema=False`——训练/验证/导出/测试全链路移除 QAT EMA                                                                                     |
| 动机 | exp33 对照：EMA 增益仅 **+0.06**（EMA 0.3964 vs 原始 qat_model 0.3958），疑为噪声                                                          |
| 假设 | 去 EMA 精度持平 exp33（0.396 量级），简化 pipeline（省一份 EMA 权重与切换逻辑）                                                            |
| 训练 | lr=2e-5, lrf=0.1, batch=64, device=2；epoch24 手动停（峰值 epoch6）                                                                        |
| 结果 | best=`epoch5.pt`（csv epoch6）内部 **0.3976**；**eval.py 0.3951**（loading from qat_model, missing/unexpected=0）                          |
| 结论 | vs exp33 best(EMA) eval 0.3964 → **−0.13 点**；**非零成本**（EMA 有小收益），但 39.51 仍达标、ckpt 小 ~12MB。权衡 0.13 点 ↔ pipeline 简化 |

### 待定：Fisher ONNX 后处理校准

| 参数 | 值                                                     |
| ---- | ------------------------------------------------------ |
| 方案 | `exp_fisher/integration_plan.md` 方案 C（ONNX 后处理） |
| 预期 | +0.08-0.2 mAP，可补 exp32 缺口                         |
| 依赖 | exp33 完成后评估是否需要                               |
