# Fisher 任务感知 IFMR 集成方案

**日期：** 2026-06-08  
**目标：** 将 `fisher_detect_refine`（检测头输出 scale 的任务加权搜索）集成到 yolo26n PT2E QAT 流水线中，在 ONNX 后处理阶段微调 3 个检测头输出的 dequantize scale，争取 +0.08–0.2 mAP。

> 理论基础见 [task_aware_ifmr_derivation.md](task_aware_ifmr_derivation.md)。核心公式：`Cost(s) = Σ_i w_i · (x_i − Q_s(x_i))²`，其中 `w_i = gate · σ'(x_f,i)² · r_component`。

---

## 1. 三种集成方式对比

| 方式 | 时机 | 改动量 | 风险 | 可行性 |
|------|------|:---:|:---:|:---:|
| **A. QAT 训练内嵌** | 每 epoch 后更新 observer scale | 大（需 Hook PT2E observer state、适配 PPQ→PT2E 图 IR） | 高 | 低。QAT 权重在变，scale 不断重搜，边际收益小 |
| **B. convert 后/export 前** | `convert_pt2e` 后修改 Q/DQ scale，再导出 ONNX | 中（需拦截 PT2E 图节点、dequantize/forward/search/update） | 中 | 中。PT2E 内部 op 操作复杂，且每次导出须重跑校准 |
| **C. ONNX 后处理（推荐）** | ONNX slim 清理后，改 3 个检测头 DQ 的 scale 初始器 | 小（3 个 initializer 替换，不碰图结构） | 零 | **高**。解耦训练/导出/校准，可对比前后 mAP，失败零风险 |

**选择方案 C**，理由：
- 校准与训练完全解耦，不增加 QAT 训练复杂度
- ONNX 图结构已稳定（slim + fix_QDQ + merge_DQ 后），只改 scale 值
- 若 Fisher 提点有效，直接保存优化后 ONNX；若无效，保留原始 ONNX，零回退成本
- 后期可扩展：同一套校准代码可应用于不同训练产出的 ONNX

---

## 2. 方案 C 实现步骤

### 步骤 1：正常导出 ONNX（现有流程，不改动）

```
export.py → build_quantized_model() → convert_pt2e() → export_onnx()
→ slim → fix_QDQ_mismatch → merge_adjacent_DQ_Q → 保存 _slim.onnx
```

产物：`qat_exp32_one2many_slim.onnx`（one2many 更接近部署需求）。

### 步骤 2：创建浮点参考 ONNX

在 slim 后的 ONNX 图上，将 3 个检测头输出路径的 `DequantizeLinear` 节点**临时替换为 Identity**（保留 scale/zp 初始器以做参考），生成纯浮点输出版本。

> 定位方法：从 ONNX 输出节点（`boxes_p3/scores_p3/*`）回溯，找到对应的 DQ 节点。

### 步骤 3：校准前向收集 logits

用 COCO val 子集（建议 64–128 batch）跑浮点参考 ONNX 推理，收集 3 个检测头输出的 float logits `x_f`。

对每个 scale 的输出张量 `x` [1, 255, H, W]：
- 计算 `gate = σ(x_obj)`（obj 通道概率，前景门控）
- 计算 `σ'(x)²`（sigmoid 导数平方，概率敏感区加权）
- 计算 `r`：box(0:3)=w_box, obj(4)=w_obj, argmax-cls(5:)=w_cls, 其余=0
- 仅保留 `gate > gate_thresh` 的前景元素（背景权重≈0 直接丢弃，省内存）
- 每元素 Fisher 权重：`w_i = gate · σ'(x_f,i)² · r_component`

### 步骤 4：搜索最优 per-tensor scale

对每个检测头输出，在 `[base_scale * s_lo, base_scale * s_hi]` 范围（建议 s_lo=0.25, s_hi=1.8, step=0.01）网格搜索：

```
s* = argmin_s Σ_i w_i · (x_i − Q_s(x_i))²
```

其中 `Q_s` 是 INT8 对称量化（qmin/qmax 从 ONNX DQ 节点获取）。

### 步骤 5：覆盖 ONNX 初始器

将搜索到的最优 scale 替换 ONNX 图中对应 `DequantizeLinear` 节点的 scale 初始器。若 scale 变化 < 2%，保持原值（避免无意义修改）。

### 步骤 6：验证与保存

- 跑 COCO val 验证修改后 ONNX 的 mAP
- 对比 Fisher 前后 mAP 变化
- 保存最终 ONNX（如 `qat_exp32_one2many_fisher.onnx`）

---

## 3. 实现参考

### ONNX scale 修改工具函数

```python
def replace_dq_scale(onnx_model, node_name, new_scale):
    """替换 ONNX 图中指定 DequantizeLinear 节点的 scale 初始器。"""
    import numpy as np
    from onnx import numpy_helper
    for node in onnx_model.graph.node:
        if node.name == node_name and node.op_type == "DequantizeLinear":
            scale_name = node.input[1]
            for init in onnx_model.graph.initializer:
                if init.name == scale_name:
                    new_tensor = numpy_helper.from_array(
                        np.array([new_scale], dtype=np.float32), name=scale_name
                    )
                    init.CopyFrom(new_tensor)
                    print(f"  {node_name}: scale {float(numpy_helper.to_array(init)[0]):.6f} -> {new_scale:.6f}")
                    return True
    return False
```

### 检测头 DQ 节点定位

从 ONNX 图输出名称回溯（如 `boxes_p3` → 最近的上游 DQ），通过 BFS 沿 `node.input` → 前驱 `node.output` 关系遍历，遇到 `DequantizeLinear` 且 scale name 可查即为目标节点。

### 简化实现路径

若不需完整的 ONNX 图修改而只需改 scale，可直接修改 slim 前原始 ONNX 中对应 DQ 的 scale 初始器（在 `_merge_adjacent_dq_q` 之后、保存 slim 之前插入 Fisher scale 优化步骤）。这是最小的代码侵入方式。

---

## 4. 预期效果

| 指标 | 当前 (exp32 best) | 预期 (Fisher 后) | 来源 |
|------|:---:|:---:|------|
| mAP50-95 | 39.82 | **≥ 39.9**（目标） | 论文 +0.4 mAP（yolov5s），yolo26n 比例换算 |
| mAP@.75 | — | 预计追赶浮点 | 论文 mAP@.75 追平 FP32 |

论文数据（yolov5s, INT8）：均匀 MSE → 0.363 mAP, +Fisher → 0.367 mAP, 检测头 FP32 上界 = 0.369。Fisher 仅改 3 个输出 scale（不改位宽、不改权重），抢回检测头量化损失 ~67%。

yolo26n 的检测头量化损耗占比与 yolov5s 可类比，保守预期 +0.08，乐观预期 +0.20。

---

## 5. 推广方向（后续实验）

当前 Fisher 仅作用于检测头输出层。论文推导可推广到**全网层**：
- 将检测头输出 Fisher 权重 `W_o` 沿网络 Gaussian-Newton 反传
- 对上游激活 `x_i`，其任务重要性 = `Σ_o W_o · (∂o/∂x_i)²`
- 用 Hutchinson 随机投影估计 Jacobian，得到每层 Fisher 权重
- 对每层 scale 做同样的加权 IFMR 搜索

这可在不额外训练的 PTQ 校准中实现，进一步抢回 backbone/neck 量化损失（检测头 FP32 上界 0.369 vs float 0.374，剩余 0.005 在上游层）。

---

## 6. 文件清单

| 文件 | 作用 |
|------|------|
| `taskaware_detect.py` | 核心实现（PPQ 版），两个入口：`fisher_detect_refine`（自动 Fisher 权重）、`taskaware_detect_refine`（手动 w_box/w_obj/w_cls） |
| `task_aware_ifmr_derivation.md` | 理论推导文档 |
| `integration_plan.md` | 本文件——PT2E QAT 集成方案 |
