---
name: yolo-qat-config-discovery
description: 为修改过网络结构或更换 PyTorch/导出环境的 YOLO26/YOLO11 PT2E QAT 模型重新发现 FX 节点，生成通用 Attention S8 与可选分类塔 U16 regional 配置，并检查训练前 FX 命中和导出后 ONNX 量化结构。用户提到节点编号变化、配置不再命中、重新生成 config-qat、分类头 16bit、Attention S8、自定义模型量化配置或结构验收时使用。
---

# YOLO QAT Config Discovery

使用模块来源和拓扑重新定位 regional 节点，不复制旧图中的 `conv2d_*`、`silu__*` 编号。

## 工作流

1. 在仓库根目录、目标训练环境中执行发现脚本。模型结构、`end2end` 和 PyTorch 版本必须与后续训练一致。
2. 通用模型使用不带 `clsU16` 的基础模板，只重新发现 Attention：

```bash
env PYTHONPATH="$PWD" \
  python .codex/skills/yolo-qat-config-discovery/scripts/discover_qat_config.py \
  --model yolo11n.yaml \
  --base-config config-qat/config_siluInU8_attnS8.json \
  --output config-qat/config_yolo11n_siluInU8_attnS8.json \
  --device cpu --imgsz 64 --expected-attention 1
```

3. 仅在模型需要分类头 U16 补偿时，改用带 `clsU16` 的组合模板：

```bash
env PYTHONPATH="$PWD" \
  python .codex/skills/yolo-qat-config-discovery/scripts/discover_qat_config.py \
  --model yolo26n.yaml --pretrained yolo26n.pt \
  --base-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --output config-qat/config_custom_siluInU8_attnS8_clsU16.json \
  --device cpu --imgsz 64 --expected-attention 2 --cls-u16 on
```

对 YOLO26 one-to-many（`end2end=False`），必须显式选择实际导出和验证的 `cv3` 分类分支；训练图即使
同时保留 two branches，也不能把 `one2one_cv3` 误作为 regional 分类塔：

```bash
env PYTHONPATH="$PWD" \
  python .codex/skills/yolo-qat-config-discovery/scripts/discover_qat_config.py \
  --model yolo26n.yaml --pretrained yolo26n.pt \
  --base-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --output config-qat/config_custom_siluInU8_attnS8_clsU16_one2many.json \
  --device cpu --imgsz 64 --expected-attention 2 --cls-u16 on --branch cv3
```

分割模型使用无 `clsU16` 的模板 `config-qat/config_yolo26nSeg_siluInU16_attnS8.json`，设置
`--cls-u16 off`，并按实际结构断言 Attention 数量。

4. 阅读脚本输出，确认每个 Attention 都包含两个 MatMul、一个缩放 Mul、一个 Softmax、QKV Conv 和
   PE Conv。启用 clsU16 时，还要确认分类分支、最后两个尺度、入口 SiLU 和同源 fan-out Conv。
5. 使用新配置从浮点权重开始 smoke 训练。不得跨图加载旧 QAT checkpoint。
6. 通过 `export.py` 导出，并执行下述 ONNX 结构检查。

## 结构检查

训练或导出前，确认现有配置中的 FX 节点仍与当前模型和环境完全一致：

```bash
env PYTHONPATH="$PWD" \
  python .codex/skills/yolo-qat-config-discovery/scripts/discover_qat_config.py \
  --model yolo26n.yaml --pretrained yolo26n.pt \
  --base-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --device cpu --imgsz 64 --expected-attention 2 --check
```

导出 `_slim.onnx` 后，按实际使用的配置自动推断 Attention 数量和分类头 U16 输出数量并验收：

```bash
python .codex/skills/yolo-qat-config-discovery/scripts/validate_qat_structure.py \
  yolo26_onnx/yolo26n_qat_slim.onnx \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --ort --expect-aligned-split-reshape 2
```

检查必须满足：每个 Attention 为 `MatMul(S8/S8 -> S8) -> Mul(S8) -> Softmax(S8) ->
MatMul(S8/S8 -> U8)`，所有匹配到的 Split/Reshape 分支 scale 和 zero-point 一致。one-to-one
检测模型还应无意外 BN/requant，且分类 score 输出位宽与配置一致。

one-to-many 的 score 已经过 Sigmoid 后回到 U8，clsU16 仅作用于 Sigmoid 前的分类 logit；验收时必须使用
`--skip-output-check`，并另行核对输出契约。当前分割 slim 图允许存在一个已知 requant（共享 P4 feature
至 box/class/mask head 时 qparams 不一致）；不得放宽导出合并阈值或手改 ONNX，应通过共享 qspec 后重训消除。

## 匹配规则

- Attention S8：按 `Attention` 模块来源分组；QKV Conv output、第一 MatMul output、缩放 Mul、Softmax 和第二 MatMul input 保持 S8，第二 MatMul output 回到全局 U8。
- 分类塔 U16 是可选策略：`--branch auto` 优先匹配 `one2one_cv3.<level>`；one-to-many 必须使用
  `--branch cv3`，不能依赖自动选择。
- 启用 clsU16 时默认选最后两个尺度。所有 SiLU output 为 U16；中间 Conv input 为 U16；最终分类 Conv input/output 为 U16。
- 最高尺度入口同时覆盖首个分类 Conv input、上游 SiLU output 及同源 fan-out 上的其他 Conv input。

## 严格约束

- 默认根据模板决定是否启用 clsU16，并配置图中所有完整 Attention 模块。已知目标图的 Attention 数量时，
  使用 `--expected-attention N` 严格断言；数量不符时不生成候选配置。
- 不按相邻编号猜测节点，不允许只替换部分旧名称。
- 输出文件不得覆盖模板；脚本会拒绝相同输入输出路径。
- 生成配置只是候选配置。成功 prepare 或导出不代表区域正确，必须检查最终 Q/DQ dtype。
- 自定义检测头若改变 `one2one/one2many` 输出契约，需要另外适配 `export.py` 和 `test.py`；配置发现时必须
  选择与实际导出契约相同的分类分支。

## 常用参数

- `--cls-last-n N`：选择分类分支最后 N 个尺度。
- `--cls-u16 auto|on|off`：默认按模板自动判断；`on` 要求模板包含 clsU16 regional 条目。
- `--expected-attention N`：可选，要求发现恰好 N 个 Attention 区域；YOLO26 当前交付图使用 2，
  YOLO11n 当前图使用 1。
- `--branch one2one_cv3|cv3|auto`：指定分类分支。
- `--report path.json`：额外保存节点角色与模块来源报告。
- `--device cuda:0`：需要复现 CUDA 特定导出图时使用，并先确认 GPU 空闲。
- `--check`：不写配置，严格检查模板中的节点名是否与当前 FX 图一致。
- `validate_qat_structure.py --quant-config path.json`：按配置检查导出后 ONNX，不依赖固定 YOLO26 节点编号。
