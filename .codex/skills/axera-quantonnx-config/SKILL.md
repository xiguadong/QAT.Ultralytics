---
name: axera-quantonnx-config
description: 从 export.py 产生的 QAT QuantONNX `_slim.onnx` 自动定位 Attention S8 区域并生成 AXERA Pulsar2 转换配置。用户需要新建或更新 axera-npu 配置、Pulsar2 layer_configs、Attention S8 手工覆盖、模型或导出环境导致 ONNX 节点名称变化，或需要检查 AXERA 转换配置是否仍匹配 QuantONNX 时使用。
---

# AXERA QuantONNX 配置

始终从待转换的精确 `_slim.onnx` 生成配置，不能从旧实验的节点名复制，也不能仅根据训练 JSON 推断 ONNX 节点名。

## 生成

训练 JSON 和 AXERA 配置使用同一量化策略文件名、不同目录。例如训练使用
`config-qat/config_siluInU8_attnS8_clsU16.json` 时，生成
`axera-npu/config_siluInU8_attnS8_clsU16.json`：

```bash
env PYTHONPATH="$PWD" \
  python .codex/skills/axera-quantonnx-config/scripts/generate_axera_config.py \
  --onnx yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --output axera-npu/config_siluInU8_attnS8_clsU16.json \
  --calibration-size 32 \
  --output-dir ./output_yolo26n_siluInU8_attnS8_clsU16
```

`--silu-data-type` 必须与训练配置的全局 SiLU input 一致：throughput/全 U8 为 `U8`，U16 基础配置或分割
配置为 `U16`。分类塔等 regional U16 由 QuantONNX 自身 Q/DQ 保留，不要把它们改写成全局 AXERA layer override。
QuantONNX 已携带量化参数，不需要校准集；生成器保留 `/path/to/dataset` 作为 Pulsar2 配置字段占位值。

YOLO11n 已验证 U8 与 SiLU input U16 两种专用配置，均有一组 Attention。具体 QKV Conv、8 个连续 S8
中间节点和第二 MatMul/PE Conv 的 S8 input 必须以生成器对本次 `_slim.onnx` 的输出为准，不在 skill 中固化
ONNX 节点名。

## 生成规则

- QKV Conv 只固定 `output_data_type=S8`。
- Reshape、Split、Q/K Transpose、首个 MatMul、Mul、Softmax、Softmax Transpose 和 V Reshape 固定输入/输出 S8。
- 第二个 MatMul 与 PE Conv 只固定 `data_type=S8`，使输出回到 QuantONNX 全局域（通常 U8）。
- 脚本会验证上述边界的实际 Q/DQ dtype；拓扑不唯一、节点缺失或 S8 域断开时必须先修正 QAT 配置和重新导出，不能手改最终 ONNX 或猜测层名。

## 转换前检查

1. 先使用 `$yolo-qat-config-discovery` 验收 `_slim.onnx` 的 Attention、Split/Reshape、BN 和 requant。
2. 检查生成配置的 `input` 等于本次要转换的 ONNX 文件，输入处理和 NPU mode 与客户工具链匹配。
3. 使用 `pulsar2 build --input <onnx> --config <config> --output_dir <dir>` 转换；完成后比较 QuantONNX 与 AXModel 精度。

分割模型可能有独立的 mask-head requant，不能用本 skill 的检测模型零 requant 规则直接判定。应从训练
qspec 排查并重新导出，不要通过放宽转换配置或手改 ONNX 掩盖问题。
