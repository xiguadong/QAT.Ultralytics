# QAT.Ultralytics

本仓库基于 [Ultralytics-8.4](https://github.com/ultralytics/ultralytics/tree/v8.4.21)，用于调试和验证 `YOLO26` 系列模型的 QAT（Quantization Aware Training）训练、导出与部署转换流程。

**使用建议：** 先确认 `qat` 到部署流程畅通。使用 `train.py` 配置小批量数据，训练 `1 epoch`；使用 `eval.py` 确认精度符合训练时的评估精度；使用`export.py` 导出 `qat_slim.onnx`；按照 [qat_deployment.md](./compile/qat_deployment.md) 进行 `axmodel` 导出。

## 精度参考

| case                 | float mAP:50~95 | float mAP:50 | axmodel mAP:50~95 | axmodel mAP:50 | err mAP:50~95 | err mAP:50 | Speed(ms) |
| -------------------- | --------------- | ------------ | ----------------- | -------------- | ------------- | ---------- | --------- |
| yolo26-one2one       | ---             | ---          | ---               | ---            | ---           | ---        | ---       |
| ptq(W8A8-Silu_U16)   | 40.24           | 55.79        | 37.83             | 53.54          | -2.41         | -2.25      | 3.583     |
| qat(W8A8-MatMul_S16) | 40.24           | 55.79        | 37.78             | 53.58          | -2.46         | -2.21      | 4.134     |
| qat(W8A8-MatMul_S8)  | 40.24           | 55.79        | 37.69             | 53.37          | -2.55         | -2.42      | 3.961     |
| yolo26-one2many      | ---             | ---          | ---               | ---            | ---           | ---        | ---       |
| ptq(W8A8-Silu_U16)   | 40.87           | 56.87        | 39.52             | 55.78          | -1.35         | -1.09      | 3.616     |
| qat(W8A8-MatMul_S16) | 40.87           | 56.87        | 39.95             | 56.69          | -0.92         | -0.18      | 4.129     |
| qat(W8A8-MatMul_S8)  | 40.87           | 56.87        | 39.89             | 56.45          | -0.98         | -0.42      | 3.962     |

注：`qat` 模型慢于 `ptq` 模型是因为 `qat` 中 `concat` 算子未共享量化参数，每一个输入输出都有量化参数，导致计算量和 `ddr swap` 高于 `qat` 模型。

测试方法：

```bash
ax_run_model -w 10 -r 100 -m xx.axmodel
```

## 安装依赖

```bash
cd ultralytics
pip install -r requirements.txt
pip install -e .
```

## 快速开始

1. 确认数据集配置可用。默认脚本依赖 `coco.yaml`，需要保证其中的数据集路径正确。
2. 运行训练：

```bash
# 单卡
python train.py

# 本仓库含自定义的qat参数，如果上述指令报错，可能使用了其他环境Ultralytics仓库，使用如下指令尝试
PYTHONPATH=/your/project/path/QAT.Ultralytics:$PYTHONPATH python train.py
```

3. 训练后评估 QAT 权重：

```bash
python eval.py
```

4. 导出 QuantONNX。运行前先确认 `export.py` 里的 `qat_weights` 指向本次训练输出：

```bash
python export.py
```

5. 使用示例图片推理验证：

```bash
python test.py
```

## 模型部署

请阅读 [qat_deployment.md](./compile/qat_deployment.md)。

## todos

- [ ] `one2one`模型精度损失小于1
- [ ] 多卡QAT
- [ ] 分割、obb、pose模型 QAT
