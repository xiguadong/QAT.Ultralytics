# 交付 Profile

## accuracy

- 配置：`config-qat/config_siluInU16_attnS8_clsU16.json`
- 全局激活/权重：U8/S8
- 全局 SiLU：input U16、output U8
- 局部 `silu__68`、`silu__97~104`：input/output U16
- 两组 Attention：第一 MatMul input/output S8，第二 MatMul input S8、output U8
- 已验证训练峰值：mAP50-95 `0.39829`
- 用途：优先保留精度余量

## throughput

- 配置：`config-qat/config_siluInU8_attnS8_clsU16.json`
- 与 accuracy 的唯一配置差异：全局 SiLU input 从 U16 改为 U8
- 局部 `silu__68`、`silu__97~104` 和 Attention S8 保持不变
- 用途：优先降低激活带宽并提高部署吞吐

## 共同约束

- `end2end=True`
- `qat_ema=False`
- `lr0=2e-5`、`lrf=0.1`
- 训练 prepare 使用 batch=2 动态 batch 图；导出必须复用训练侧 `prepare_pt2e_qat_model`
- 导出模型必须为 BN=0、requant=0
- AXERA NPU 不支持 Split 分支独立量化参数，slim 后处理必须启用 Split/Reshape 量化范围对齐
