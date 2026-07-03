# Validator 修复 & 配置对齐

**日期：** 2026-05-06  
**目标：** 修复 qat_validate 精度虚高问题，并与参考仓库配置对齐

---

## 问题发现

`qat_validate=True` 时 results.csv 记录的 mAP50-95 为 **0.527**，远高于 float baseline 0.392，明显异常。

**根本原因：** `validator.py` 第203行，当 `pt2e_model is trainer.qat_model` 时，模型保持 train 模式（`model.float()` 但不切 eval），BN 使用 batch statistics 而非 running stats，导致精度虚高。

---

## 修改内容

### 1. `ultralytics/engine/validator.py`

**修复 qat_model 在 val 时未切换到 eval 模式的 bug。**

```diff
-            if pt2e_model is not None:
-                # Prepared PT2E QAT graphs can fail after switching to eval in torch 2.6 on YOLO26 heads.
-                # Keep the training graph for `trainer.qat_model` validation and only rewrite true exported eval graphs.
-                if pt2e_model is getattr(trainer, "qat_model", None):
-                    model = model.float()
-                else:
-                    model = self._prepare_pt2e_model_for_eval(model.float())
+            if pt2e_model is not None:
+                model = self._prepare_pt2e_model_for_eval(model.float())
```

修复后 epoch 0 prepared 模型精度：**0.382**（符合预期，接近参考仓库 INT8 结果 0.386）。

### 2. `config.json`

与参考仓库 `ax_quantizer.py` 默认值对齐：

```diff
-"act_observer": "histogram",
-"weight_observer": "per_channel",
+"act_observer": "moving_avg",
+"weight_observer": "moving_avg_per_channel",
```

### 3. `ultralytics/cfg/default.yaml`

```diff
-erasing: 0.4
+erasing: 0.0
```

参考仓库明确设为 0，QAT 阶段不需要随机擦除增强。

### 4. `train.py`

```diff
-lrf=0.2,
+lrf=0.1,
-name="exp13-yolo11n-histogram",
+name="exp14-yolo11n-moving-avg",
```

---

## 验证结果

| 指标                             | 修复前      | 修复后        |
| -------------------------------- | ----------- | ------------- |
| qat_validate mAP50-95（epoch 0） | 0.527（假） | 0.382（真实） |
| float baseline                   | 0.392       | 0.392         |
| 参考仓库 INT8 目标               | 0.386       | —             |

---

## 当前训练

**exp14**（PID 2299476）：moving_avg observer，修复后 validator，50 epochs。  
日志：`runs/detect/exp14-yolo11n-moving-avg.log`
