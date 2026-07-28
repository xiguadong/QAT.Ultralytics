# 仓库协作说明

- 所有回复使用中文。在仓库根目录工作，先激活项目兼容的 PyTorch 2.6 QAT 环境，并设置 `PYTHONPATH="$PWD"`。
- 本仓库用于 YOLO26 PT2E QAT。优先修复检测和分割训练问题，不要中断其他正在运行的训练任务。
- 精度目标：YOLO26n `end2end=True` 浮点 40.2、QAT >= 39.2；YOLO26s 浮点 47.8、QAT >= 47.3。
- 精度目标：YOLO26n `end2end=False` 浮点 40.9、QAT >= 39.9；YOLO26s 浮点 48.6、QAT >= 48.0。
- 最终 YOLO26 检测交付只使用两个 profile：`accuracy=config-qat/config_siluInU16_attnS8_clsU16.json`，`throughput=config-qat/config_siluInU8_attnS8_clsU16.json`；自定义模型或量化边界训练使用 `train_qat.py --quant-config <json>`，显式配置优先于 profile。
- 网络结构、PyTorch 版本或导出环境变化后，使用 `$yolo-qat-config-discovery` 重新定位分类塔 U16 和 Attention S8 节点，不得沿用旧 FX 编号。
- Attention S8 是通用策略；分类塔 U16 是可选补偿。YOLO11 默认使用不带 `_clsU16` 的基础配置。
- 训练统一使用 `train_qat.py`；评估统一使用 `eval.py`；QAT 导出必须使用 `export.py`；图片验证使用 `test.py`。
- 导出失败应检查 checkpoint、量化配置和图结构是否一致。不得放宽 `state_dict` 加载或修改 `ultralytics/utils` 来迁就异常权重。
- 最终 QuantONNX 必须满足：BN=0、`_requant` Identity=0、Attention MatMul 为 S8、两处 Split/Reshape 量化参数符合 AXERA 约束。
- 修改 PT2E prepare、BN、quantizer 或导出后处理后，运行 QAT 定向测试，并核对 exported/prepared/QAT 数值及 BN `momentum=0.03`、`eps=0.001` 未漂移。
- AXERA 部署文件位于 `axera-npu/`。
- 工作区可能包含用户修改。不要回退、覆盖或清理与当前任务无关的变更。
