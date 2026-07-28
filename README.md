# QAT.Ultralytics

本仓库基于 [Ultralytics-8.4](https://github.com/ultralytics/ultralytics/tree/v8.4.21)，提供
`YOLO26` 和 `YOLO11` 系列 PT2E QAT 的训练、QuantONNX 导出和 AXERA 部署兼容处理。

## 精度结果

### 检测

以下为当前 AXERA NPU 实测结果：

| 模型 | `end2end` | 配置 | FP32 mAP50-95 | FP32 mAP50 | QAT mAP50-95 | QAT mAP50 | Err mAP:50~95 | Err mAP:50 | Speed(ms) |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26n | `true` | ptq(w8a8_siluInU16) | 40.24 | 55.79 | 37.83 | 53.54 | -2.41 | -2.25 | 3.613 |
| YOLO26n | `true` | `config_siluInU16_attnS8_clsU16.json` | 40.24 | 55.79 | 39.61 | 55.63 | -0.63 | -0.16 | 3.761 |
| YOLO26n | `true` | `config_siluInU8_attnS8_clsU16.json` | 40.24 | 55.79 | 39.39 | 55.37 | -0.85 | -0.42 | 3.656 |
| YOLO26n | `false` | ptq(w8a8_siluInU16) | 40.87 | 56.87 | 39.52 | 55.78 | -1.35  | -1.09 | 3.616 |
| YOLO26n | `false` | `config_siluInU8_attnS8_clsU16_one2many.json` | 40.87 | 56.87 | 39.97 | 56.57 | -0.9 | -0.3 | 3.647 |
| YOLO11n | `None` | ptq(w8a8_siluInU16) | 39.4 | 55.3 | 38.8 | 54.55 | -0.6 | -0.75 | 3.934 |
| YOLO11n | `None` | `config_yolo11n_siluInU8_attnS8.json` | 39.4 | 55.3 | 38.45 | 54.46 | -0.95 | -0.84 | 3.814 |
| YOLO11n | `None` | `config_yolo11n_siluInU16_attnS8.json` | 39.4 | 55.3 | 38.84 | 54.86 | -0.56 | -0.44 | 3.953 |

性能：AX650N NPU1 模式，采用 `ax_run_model -w 10 -r 100 -m <model>.axmodel` 测得。

注：`end2end=true` 为 `one2one`模型，后处理无`nms`；`end2end=false`为`one2many`模型，后处理需`nms`。`clsU16`指的是P4、P5分类头`data_type=U16`，权重还是`s8`。

### 分割

以下为 COCO-Seg 验证集上的 QAT 训练内最优结果。`seg_best` 已完成 QuantONNX 导出与 ORT 推理。

| 模型 | `end2end` | 配置 | FP32 Box mAP50-95 | QAT Box mAP50-95 | Box 误差 | FP32 Mask mAP50-95 | QAT Mask mAP50-95 | Mask 误差 | epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n-seg | `true` | `config_yolo26nSeg_siluInU16_attnS8.json` | 39.75 | 38.86 | -0.90 | 33.86 | 33.20 | -0.66 | 44（Box 峰值） |
| YOLO26n-seg | `true` | `config_yolo26nSeg_siluInU16_attnS8.json` | 39.75 | 38.83 | -0.92 | 33.86 | 33.32 | -0.54 | 15（Mask 峰值） |

注：分割模型未上板测试，该指标为训练指标。

## QAT 量化配置

已验证的配置组合、SiLU/Attention/分类头量化边界，以及自定义网络重新发现节点的流程统一见
[AXERA QAT Quantizer 配置说明](./axera-npu/quantizer_configuration.md#31-qat-配置与已验证组合)。JSON 文件保留在
`config-qat/`；`train_qat.py` 的 `--profile accuracy|throughput` 仅对应 YOLO26n 的两个交付 profile，YOLO11、
分割和自定义网络必须显式传入 `--quant-config <json>`。

详细训练、导出和 ONNX 验收流程见[$skill-yolo26-qat-delivery](.codex/skills/yolo26-qat-delivery/SKILL.md)。
网络结构或导出环境变化后，使用[$skill-yolo-qat-config-discovery](.codex/skills/yolo-qat-config-discovery/SKILL.md)
重新发现 regional 节点。

## 快速开始

1. 安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

2. 确认数据集配置可用。默认脚本依赖 `coco.yaml`，需要保证其中的数据集路径正确。
3. 先做 smoke：

```bash
env PYTHONPATH="$PWD" \
  python train_qat.py \
  --profile throughput --device 0 --epochs 1 --batch 2 --imgsz 64 \
  --workers 0 --fraction 0.01 --name qat-delivery-smoke --exist-ok
```

4. 正式训练：

```bash
env PYTHONPATH="$PWD" \
  python train_qat.py \
  --profile throughput --device 0
```

YOLO11n 使用专用 JSON 显式启动训练，不使用仅面向 YOLO26n 的 `--profile`：

```bash
env PYTHONPATH="$PWD" \
  python train_qat.py \
  --task detect --model yolo11n.yaml --pretrained yolo11n.pt --data coco.yaml \
  --quant-config config-qat/config_yolo11n_siluInU8_attnS8.json \
  --device 0 --name yolo11n-qat-siluInU8-attnS8
```

单卡入口支持检测和分割；未提供 `--quant-config` 或 `--profile` 会直接报错，避免无意回落到某个交付配置。
当前已验证的 YOLO26n-seg 训练参数如下：

```bash
env PYTHONPATH="$PWD" \
  python train_qat.py \
  --task segment --model yolo26n-seg.yaml --pretrained weights/yolo26n-seg.pt \
  --data coco-seg.yaml --quant-config config-qat/config_yolo26nSeg_siluInU16_attnS8.json \
  --epochs 50 --batch 64 --imgsz 640 --workers 8 --fraction 1.0 \
  --lr0 2e-5 --lrf 0.1 --no-qat-ema --end2end --save-period 1 \
  --device 0 --name yolo26n-seg-qat
```

正式训练默认使用完整数据集（`fraction=1.0`）；`--fraction 0.01` 仅用于前述 smoke，不应用于精度验收。

实验性多 GPU QAT 使用独立入口。`--batch` 是全局 batch，必须能被 GPU 数量整除；先用小数据 smoke
确认显存、DDP 构图和 checkpoint 导出正常，再进行正式训练：

```bash
env PYTHONPATH="$PWD" \
  python train_gpus.py \
  --profile throughput --devices 0,1 \
  --epochs 1 --batch 4 --imgsz 64 --workers 0 --fraction 0.01 \
  --name qat-ddp-smoke --exist-ok
```

两卡正式检测训练示例。`--batch 64` 是全局 batch，每个 rank 使用 32；使用自定义模型时替换
`--model`、`--pretrained`、`--data` 与 `--quant-config`，并先重新发现 regional 节点：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python train_gpus.py \
  --task detect --model yolo26n.yaml --pretrained weights/yolo26n.pt --data coco.yaml \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --devices 0,1 --epochs 50 --batch 64 --imgsz 640 --workers 4 --fraction 1.0 \
  --lr0 2e-5 --lrf 0.1 --no-qat-ema --end2end --save-period 1 \
  --name yolo26n-qat-throughput-ddp
```

`--devices` 接受任意数量不少于 2 的、不重复的单机 GPU 编号，例如 `--devices 0,1,2,3`。启动进程数
等于 GPU 数量，每个 rank 的 batch 为 `--batch / GPU 数量`；`--workers` 同样按每个 rank 计算。三卡时
不能使用全局 batch 64，应改为 60、63、96 等可整除数值。

多卡入口同样可用 `--quant-config path/to/custom.json` 替代 `--profile`，显式配置优先；目前只完成
检测任务的 DDP 验证，分割任务请使用单卡 `train_qat.py`。

多 GPU QAT 会在每个 rank 独立构建相同 PT2E 图，并使用 DDP 同步可训练参数。目前已完成两张 GPU 的
64 和 640 输入 smoke、完整 COCO 单轮训练与验证、checkpoint 严格加载及 ONNX 结构检查；三卡及更多
卡数、多机训练尚未验证。QAT DDP 验证由 rank 0 使用完整非分布式验证集执行，再向其他 rank 广播指标，
避免检测统计在分布式验证收尾时发生 collective 不同步。
Observer buffer 依赖 DDP 默认广播，正式交付前仍需与单卡结果对齐 `convert_pt2e` 精度和最终 ONNX 结构。

Quantizer 配置能力及自定义网络注意事项见[quantizer_configuration.md](./axera-npu/quantizer_configuration.md)。

5. 评估真实 Q/DQ 精度。交付判定以 `convert` 为准，`qat` 用于对照训练内 fake-quant：

```bash
env PYTHONPATH="$PWD" \
  python eval.py convert \
  --ckpt runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json --device cuda:0
```

统一评估入口还提供 `float`、`qat`、`onnx`、`segment`、`ptq` 和兼容用的
`onnx-one2many` 子命令，使用 `python eval.py <mode> --help` 查看参数。

分割 QAT checkpoint 使用 `segment` 模式。评估会从 checkpoint 恢复训练时的 `overlap_mask` 和
`mask_ratio`，QAT state 必须与当前 prepared graph 严格匹配：

```bash
env PYTHONPATH="$PWD" \
  python eval.py segment \
  --ckpt runs/segment/yolo26n-seg-qat/weights/best.pt \
  --model yolo26n-seg.yaml --pretrained weights/yolo26n-seg.pt \
  --quant-config config-qat/config_yolo26nSeg_siluInU16_attnS8.json \
  --data coco-seg.yaml --device 0 --batch 16
```

6. 显式指定配置和权重导出 QuantONNX：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python export.py \
  --task detect --model yolo26n.yaml --pretrained yolo26n.pt \
  --qat-weights runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --out yolo26_onnx/yolo26n_qat_throughput.onnx \
  --device cuda:0 --imgsz 640 640 --end2end true
```

7. 验收 slim ONNX：

```bash
python \
  .codex/skills/yolo-qat-config-discovery/scripts/validate_qat_structure.py \
  yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --ort --expect-aligned-split-reshape 2
```

8. 对图片或图片目录执行 QAT 推理并保存绘制结果：

```bash
# QAT checkpoint：自动读取 checkpoint 中记录的 quant config
env PYTHONPATH="$PWD" \
  python test.py \
  --model runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --source bus.jpg --device cpu

# 六输出 QuantONNX
env PYTHONPATH="$PWD" \
  python test.py \
  --model yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --source bus.jpg --device cpu
```

绘制图片默认保存到 `runs/predict/qat-test/`。使用 `--save-txt --save-conf` 可同时保存 YOLO 格式检测结果。

分割 checkpoint 和 QuantONNX 使用相同入口，增加 `--task segment`。输出图同时绘制检测框和实例 mask：

```bash
env PYTHONPATH="$PWD" \
  python test.py --task segment \
  --model runs/segment/yolo26n-seg-qat/weights/best.pt \
  --model-yaml yolo26n-seg.yaml --pretrained weights/yolo26n-seg.pt \
  --source bus.jpg --device 0

env PYTHONPATH="$PWD" \
  python test.py --task segment \
  --model yolo26_onnx/yolo26n_seg_qat_slim.onnx \
  --model-yaml yolo26n-seg.yaml --pretrained weights/yolo26n-seg.pt \
  --source bus.jpg --device cpu
```

## 模型部署
请阅读 [qat_deployment.md](./axera-npu/qat_deployment.md)。
