# 任务感知 IFMR 量化 Cost 的推导

> 配套实验与结论见 [task_v1.md](task_v1.md)。实现：`quant/scripts/taskaware_detect.py`。
> 一句话结论：标准 IFMR 最小化**均匀**重建 MSE，对检测 mAP 是错的目标；正确的 cost 是用**任务损失曲率**加权的
> 重建误差，对检测头输出可化简为 `w_i = gate·σ'(x_f,i)²·r_i`，实测 yolov5s minmax+calib4 **0.363→0.367（纯 INT8）**。

---

## 0. 问题与动机

INT8 PTQ 给每个张量选一个量化 scale `s`，做 `Q_s(x)=clip(round(x/s)+z, qmin,qmax)`，反量化 `x̂=(Q_s(x)-z)·s`。
**标准 IFMR / MSE 校准**搜索 `s` 最小化重建误差：
```
s* = argmin_s  Σ_i (x_i − x̂_i)²            （均匀加权，每个元素同权）
```

实测发现：在 yolov5s 上，minmax / mse / IFMR / EasyQuant 给出的 scale 差异很大（mse 比 minmax 裁 37%），
**但 mAP 几乎不变**。原因（实验 E3）：检测头输出 `[1,3,H,W,85]` 的量化误差 **94.9% 在 80 个类别通道、
99.85% 在背景格子**——而 mAP 只取决于**前景格子的 obj/box/argmax 类别**。
所以均匀 MSE 把几乎全部优化预算花在了**对 mAP 无关**的数值上 ⇒ 降误差不涨 mAP。

**目标**：推导一个 cost，使 `argmin_s Cost(s)` 给出的 scale 最大限度保住 mAP。

---

## 1. 一般推导：任务损失曲率加权的重建误差

设 `L(x)` 是任务质量的可微代理（越小=越接近浮点模型的检测结果）。量化引入扰动 `δx = x̂ − x`。
对 `L` 在浮点点 `x` 处二阶泰勒展开：
```
ΔL = L(x+δx) − L(x) = ∇L(x)·δx + ½ δxᵀ H δx + O(δx³)，   H = ∂²L/∂x²
```

量化舍入误差 `δx` 近似**零均值、逐元素独立**（dither 假设），故一阶项期望约为 0：`E[∇L·δx] ≈ 0`。
退化由二阶项主导，取 Hessian 对角近似 `H_ii`：
```
E[ΔL] ≈ ½ Σ_i H_ii · E[δx_i²] = ½ Σ_i H_ii · (x_i − x̂_i)²
```

⇒ **任务感知量化 cost：**
```
┌─────────────────────────────────────────────┐
│  Cost(s) = Σ_i  H_ii · (x_i − Q_s(x_i))²      │   H_ii = ∂²L/∂x_i²
└─────────────────────────────────────────────┘
```
即"用任务损失曲率 `H_ii` 加权的重建误差"。**标准 IFMR 是 `H_ii ≡ 1` 的特例**。
这与 Hessian-aware 量化（HAWQ / BRECQ / AdaRound）同源——所以正确的做法不是手调权重，而是算 `H_ii`。

---

## 2. 检测头输出的 H_ii：化简为 gate·σ'²·r

检测头输出是 logits 张量。记每个 anchor `a` 的分量：box(4) + obj(1) + cls(80) = 85。
解码后的检测量 `D_{a,k}`（box 坐标、obj 概率、cls 概率）才是 mAP 关心的，且都经过 **sigmoid**：
- `p_obj = σ(t_obj)`，`p_cls = σ(t_cls)`，box 也走 σ-based 解码（`(2σ−0.5)`、`(2σ)²`）。

用"保住浮点检测"的保真代理（按相关性 `r` 加权的检测量距离）：
```
L = ½ Σ_a Σ_k r_{a,k} · ( D_{a,k}(x) − D_{a,k}(x_f) )²
```
在 `x = x_f` 处，这是平方和，Gauss-Newton ⇒ Hessian 对角：
```
H_ii ≈ Σ_{a,k} r_{a,k} · ( ∂D_{a,k}/∂x_i )²
```
每个输出 logit `x_i` 只通过 sigmoid 进入一个检测量，`∂D/∂x_i = σ'(x_i)·(常数因子)`，于是：
```
┌─────────────────────────────────────────────┐
│   H_ii ≈ r_i · σ'(x_{f,i})²,   σ'(z)=σ(z)(1−σ(z))   │
└─────────────────────────────────────────────┘
```

### 关键洞察（灵光一闪）
**量化作用在 logits 上，但 mAP 取决于 sigmoid 之后的概率。** 所以一个 logit 的重要性天然带 `σ'(logit)² = [p(1−p)]²`：
- `p ≈ 0.5`（决策边界）的 logit **极其重要**——微小 logit 扰动会翻转概率/检测；
- `p ≈ 0/1`（饱和）的 logit **几乎无关**——扰动被 sigmoid 压平。

这个因子不是设的，是从"量化在 logit、任务在概率"**自动**推出来的。

### 相关性权重 r_i（同样自动归零无关项）
```
r_i = gate_a · 分量系数
  gate_a = σ(t_obj,a)          前景门控：背景 obj 概率低 → gate≈0 → 该 anchor 全部元素权重≈0（自动丢背景）
  分量： box(0:4)、obj(4)、argmax 类别 → 有权；
         非 argmax 的 79 个类别 → r=0（它们不影响 argmax/score，自动丢无关类）
```

### 合并：每元素权重（全部来自推导，零手调）
```
┌──────────────────────────────────────────────────────┐
│   w_i = gate_a · σ'(x_{f,i})² · r_component             │
│   Cost(s) = Σ_i w_i · (x_i − Q_s(x_i))²                 │
└──────────────────────────────────────────────────────┘
```
它同时、自动地实现了：**背景归零、无关 79 类归零、概率敏感区加权**——正是 mAP 在乎的子集。

---

## 3. 实现与结果

`quant/scripts/taskaware_detect.py::fisher_detect_refine`：
1. 浮点图前向，取 3 个检测头输出 `x_f`（仅收集前景 anchor-location 的元素，背景权重≈0 直接丢，省内存）。
2. 按上式算每元素权重 `w_i`（`gate=σ(obj_f)`，`σ'²`，box/obj/argmax-cls 有权）。
3. 搜索每个检测头输出的 per-tensor scale：`s* = argmin_s Σ_i w_i (x_i − Q_s(x_i))²`。
4. 只改 `output_quantization_config.scale`，**保持量化态 ACTIVATED（仍 INT8）**。

**效果**：检测头输出 scale 被砍半（如 0.107→0.052）——前景相关元素值域比被大类别 logit 主导的整体小得多，
Fisher 最优 scale 更小 → 给相关元素更细分辨率、把无关的大类别值裁掉（与手调 w_cls 把 scale 调大的方向相反）。

**mAP（yolov5s, minmax+calib4, COCO val5000）**：
| 配置 | mAP@.5:.95 | mAP@.75 |
|---|---|---|
| 全量化（均匀 MSE） | 0.363 | 0.390 |
| **+ Fisher cost（仍 INT8）** | **0.367** | **0.397** |
| 检测头输出 FP32（上界） | 0.369 | 0.397 |
| float | 0.374 | — |

**+0.004（~4σ 显著），不改位宽，抢回检测头量化损失 ~67%；mAP@.75 框定位追平 FP32。** 已验证纯 INT8（输出 state=ACTIVATED、256 个量化值）。

---

## 4. 推广到上游层（Gauss-Newton 反传）

检测头输出 FP32 上界是 0.369，剩余 0.369→0.374 在**上游激活**。把 `H_ii` 沿网络反传：
对上游激活 `x` 的元素 `x_i`，其任务重要性 = 它经网络对各检测头输出元素 `o` 的影响、再乘 `o` 的任务权重 `W_o`：
```
w_i ≈ Σ_o W_o · ( ∂o/∂x_i )²     （Gauss-Newton；W_o = 第2节的检测头输出权重）
```
可用反传/雅可比估计（Hutchinson 随机投影）得到逐元素 Fisher 权重，再代入同一个加权 IFMR 搜索。
这把"任务感知 IFMR"从检测头扩展到**全网**，是抢回剩余损失、形成通用方法的路径（实现中）。

---

## 参考脉络
Hessian-aware PTQ：AdaRound（逐层二阶重建）、BRECQ（块级 GN-Hessian）、HAWQ（Hessian 谱混合精度）。
本文的特化点：**针对检测任务，把输出 Hessian 化简为 `gate·σ'²·r`，揭示 sigmoid 导数因子**，并验证它对 mAP 有效。
