# exp34 — 移除 QAT 全链路 EMA

**日期：** 2026-07-01
**分支：** `qat`
**基线：** exp33（S8 matmul, e2e=False, best mAP50-95 = 39.90 / eval.py 复现 0.3964）

---

## 一、动机

对 exp33 `best.pt` 做 EMA vs 原始权重对照评估（同一 eval.py pipeline，config_matmul_s8.json，全量 COCO val）：

| 权重 | mAP50-95 | mAP50 |
|------|:---:|:---:|
| `qat_ema`（EMA） | **0.3964** | 0.5599 |
| `qat_model`（原始） | **0.3958** | 0.5609 |
| EMA 增益 | **+0.0006（+0.06 点）** | −0.0010 |

- EMA 与原始权重差异其实很大：3951 个张量中 **2178 个不同**，463 个量化 scale 中 **462 个不同**。
- 但在 QAT 低 lr（2e-5）微调下，两者收敛到几乎同一精度，落到 mAP 上拉不开。
- 结论：EMA 在本 run 是"锦上添花但花极小"（+0.06 点，噪声量级）。

## 二、假设

去掉 QAT EMA 后精度持平 exp33（0.396 量级，≥ 39.2 达标），并简化 pipeline：
- 训练不再维护 EMA 权重（省显存/一份 state）
- 验证/导出/测试不再需要 EMA↔原始 的切换与优先级逻辑

## 三、改动（新增 `qat_ema` 开关，default.yaml 默认 True 保持旧行为）

| 环节 | 改动 |
|------|------|
| default.yaml | 新增 `qat_ema: True` |
| trainer.py | 惰性创建 EMA 加 `getattr(self.args, "qat_ema", True)` 守卫；`qat_ema=False` 时全程不建/不更新 |
| 验证 | 已有 `if self.qat_ema is not None` 守卫，ema=None 时直接验 `qat_model` |
| model.py | resume 重建 EMA 加同款守卫 |
| eval.py / eval-seg.py / export.py | 修 None 回落 bug：`ckpt.get("qat_ema") or ckpt.get("qat_model")`，使 exp34（ckpt 内 `qat_ema=None`）自动回落到 `qat_model` |

向后兼容：exp33 ckpt 含真实 EMA → 仍走 EMA，数值不变。

## 四、训练配置

```
name=exp34-yolo26n-S8matmul-noEMA
qat=True, qat_config=config_matmul_s8.json, qat_validate=True, qat_ema=False, end2end=False
epochs=50, batch=64, imgsz=640, device=2, fraction=1, lr0=2e-5, lrf=0.1
```

## 五、验收标准

1. 开训自检：日志无 EMA 创建，每 epoch 验证 qat_model，results.csv 正常写 mAP。
2. best mAP50-95 vs exp33（0.3964）Δ 在 ±0.1 点内 → **确认去 EMA 零成本**。
3. eval.py 复跑 exp34 best.pt：打印 `loading from qat_model`，missing=0/unexpected=0，scale 零漂移。
4. 回归：改动后 eval.py 复跑 exp33 旧 ckpt 仍 `loading from qat_ema`、数值不变。

## 六、结果（待回填）

| 项 | 值 |
|----|---|
| best 权重 | `epoch5.pt` ≡ `best.pt`（md5 相同）= csv epoch6，训练内部峰值 **0.3976** |
| eval.py 复现 | **0.3951**（mAP50=0.5584）；`loading from qat_model`、missing=0/unexpected=0、内部 val 0.3976 判 MATCH |
| vs exp33 Δ | exp33 best(EMA) eval **0.3964** → exp34(无EMA) **0.3951**，**−0.13 点** |
| 结论 | **去 EMA 非零成本，约掉 0.13 点**（EMA 有小而真实收益，与 exp33 同-run 对照 +0.06 同向）；但 0.3951=39.51 **仍稳过 39.2**，且 ckpt 小 ~12MB、pipeline 简化。权衡 0.13 点精度 ↔ 更简单流程 |

> 备注：exp34 训练到 epoch24 手动停止（峰值早在 epoch6，参考 exp33 亦 epoch9 见顶后走平，续训预计无提升）。
> 权重实际存于 `/home/heqi/project/QAT.Ultralytics/runs/detect/runs/detect/exp34-yolo26n-S8matmul-noEMA/weights/`
> （从该 cwd 相对 project 启动所致，非主项目目录）。
> 代码链路验证通过：训练不建 EMA、验证走 qat_model、ckpt 内 qat_ema=None、eval.py/export.py 的 `or` 回落正确命中 qat_model。
