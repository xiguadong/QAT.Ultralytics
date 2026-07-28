# CLAUDE.md

本文件说明在此仓库中进行 YOLO26 PT2E QAT 开发、训练和交付时必须遵循的约定。

## 基本约定

- 所有回复使用中文。
- 工作目录为仓库根目录。
- 使用项目兼容的 PyTorch 2.6 QAT 环境，文档命令假设该环境已经激活。
- 执行仓库脚本时设置 `PYTHONPATH="$PWD"`。
- 不要停止或干扰其他正在运行的训练进程；启动训练前先确认目标 GPU 空闲。
- 不得回退用户已有修改，不得用宽松 checkpoint 加载或模型代码补丁掩盖量化图错误。

## 精度目标

| 模型    | 模式            | 浮点基线 | QAT 目标 |
| ------- | --------------- | -------: | -------: |
| YOLO26n | `end2end=True`  |     40.2 |  >= 39.2 |
| YOLO26s | `end2end=True`  |     47.8 |  >= 47.3 |
| YOLO26n | `end2end=False` |     40.9 |  >= 39.9 |
| YOLO26s | `end2end=False` |     48.6 |  >= 48.0 |

## 交付 Profile

最终 YOLO26n 检测模型只使用以下配置：

| Profile      | 配置                                             | 量化边界                                                      |
| ------------ | ------------------------------------------------ | ------------------------------------------------------------- |
| `accuracy`   | `config-qat/config_siluInU16_attnS8_clsU16.json` | 全局 SiLU input U16，Attention S8，分类塔局部 U16             |
| `throughput` | `config-qat/config_siluInU8_attnS8_clsU16.json`  | 全局 SiLU input/output U8，保留 Attention S8 和分类塔局部 U16 |

不要恢复 exp57 之前的一次性训练脚本和候选量化配置。profile 或图结构变化后，应从浮点预训练权重重新训练，不能跨 profile 续训。
图结构、PyTorch 版本或导出环境变化后，先使用 `$yolo-qat-config-discovery` 按模块来源和拓扑重新生成 regional 节点配置。
Attention S8 使用不带 `_clsU16` 的通用基础配置；YOLO26 当前交付按需叠加分类塔 U16，YOLO11 默认不启用该补偿。

## 统一入口

### 训练

```bash
env PYTHONPATH="$PWD" \
  python train_qat.py \
  --profile throughput --device 0
```

正式训练前先使用小图片尺寸、少量数据和 1 epoch 做 smoke。当前交付 profile 使用 `end2end=True`、`qat_validate=True`、`qat_ema=False`。

`--profile` 仅是 YOLO26 交付快捷方式。自定义网络、YOLO11、分割模型或自行调整量化区域时，必须显式传入
`--quant-config <json>`；该参数优先于 `--profile`。单卡 `train_qat.py` 支持 `detect` 和 `segment`，多卡
`train_gpus.py` 当前只验收 `detect`：

```bash
env PYTHONPATH="$PWD" \
  python train_qat.py \
  --task segment --model yolo26n-seg.yaml --pretrained weights/yolo26n-seg.pt \
  --data coco-seg.yaml --quant-config config-qat/config_yolo26nSeg_siluInU16_attnS8.json \
  --device 0 --name yolo26n-seg-qat
```

### 评估

统一使用 `eval.py` 子命令：

```bash
env PYTHONPATH="$PWD" \
  python eval.py convert \
  --ckpt runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json --device cuda:0
```

- `convert`：真实 Q/DQ 精度，作为交付判定依据。
- `qat`：prepared fake-quant 对照。
- `onnx`：六输出 one2one QuantONNX 精度。
- `float`：浮点基线。
- `segment`、`ptq`、`onnx-one2many`：专项或兼容评估。

### 导出

QAT checkpoint 必须通过 `export.py` 导出：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python export.py \
  --task detect --model yolo26n.yaml --pretrained yolo26n.pt \
  --qat-weights runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --out yolo26_onnx/yolo26n_qat_throughput.onnx \
  --device cuda:0 --imgsz 640 640 --end2end true
```

默认不导出 `.pth`；只有明确需要时使用 `--export-pth`。导出失败说明 checkpoint、配置或量化图存在问题，禁止修改模型加载逻辑强行导出。

### 图片推理

`test.py` 支持 QAT checkpoint 和六输出 QuantONNX，可接受单张图片、目录或 glob，并保存绘制结果：

```bash
env PYTHONPATH="$PWD" \
  python test.py \
  --model yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --source image.jpg --device cpu --save-txt --save-conf
```

输入 QAT `.pt` 时会按 checkpoint 的 quant config 重建 prepared graph，不能用普通 `YOLO(best.pt)` 代替。

## ONNX 验收

使用 `.codex/skills/yolo26-qat-delivery/scripts/validate_onnx.py` 验收 `_slim.onnx`。最终模型必须满足：

- ONNX checker 和 ONNX Runtime 加载通过。
- `BatchNormalization=0`。
- 名称包含 `_requant` 的 Identity 数量为 0。
- 四个 Attention MatMul 输入为 S8，第一组输出为 S8，第二组输出为 U8。
- 两处 Split/Reshape 使用 AXERA 后端要求的对齐量化参数。

Split/Reshape 对齐是 `export.py` 的 AXERA 兼容后处理，可以保留；不要手工修改最终 ONNX 来掩盖训练图问题。

## PT2E 与量化约束

- `ultralytics/utils/qat_utils.py`：PT2E QAT prepare 入口。
- `ultralytics/utils/pt2e_bn_patch.py`：保留 YOLO BN 参数。
- `ultralytics/utils/ax_quantizer.py`：主要 AXERA PT2E quantizer。
- `ultralytics/utils/ax_quantizer_lsq.py`：LSQ 变体，不属于当前交付 profile。
- `ultralytics/utils/ax_quantizer_utils.py`：量化配置和 qspec 工具。

涉及 `torch.export.export_for_training`、`prepare_qat_pt2e` 或 train/eval 切换时，必须检查：

- exported、prepared 和关闭 observer/fake-quant 后的数值对齐。
- BatchNorm `momentum=0.03`、`eps=0.001` 在导出和 prepare 后未变为 torch 默认值。
- QAT checkpoint 与 quant config 的 observer 节点完全匹配。

## 回归测试

```bash
env PYTHONPATH="$PWD" \
  python -m pytest \
  tests/test_pt2e_bn_patch.py \
  tests/test_qat_observer_config.py \
  tests/test_export_qat_prepare.py \
  tests/test_qat_export_postprocess.py
```

修改训练主流程时，再运行 `tests/test_qat_engine.py` 对应的 slow smoke。提交前运行 `git diff --check`。

## AXERA 部署

部署脚本、Pulsar2 配置和说明统一位于 `axera-npu/`，入口为 `axera-npu/qat_deployment.md`。不要恢复已删除的 `compile/` 目录。

## 本地资产

以下目录只用于本地实验和产物保存，不进入源码交付：

- `runs/`
- `weights/`
- `yolo26_onnx/`
- `todos/`
- `todos-seg/`

源码、最终量化配置、`axera-npu/`、测试和 `.codex/skills/yolo26-qat-delivery/` 应保持可跟踪。
