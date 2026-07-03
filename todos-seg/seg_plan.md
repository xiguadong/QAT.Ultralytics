# yolo26n-seg QAT 训练规划

**Created:** 2026-06-10  
**Updated:** 2026-06-11  
**目标：** 跑通 yolo26n-seg 的 QAT 全流程（训练→评估→导出 ONNX）

---

## 已知基准

| 项目                       |                      数值                      |
| -------------------------- | :--------------------------------------------: |
| yolo26n-seg float box mAP  |            **39.6**（end2end=True）            |
| yolo26n-seg float mask mAP |            **33.9**（end2end=True）            |
| 训练集                     |             COCO 2017, 118,287 张              |
| 配置                       |       W8A8 + S8 matmul（对标检测 exp33）       |
| 训练                       | 50 epoch, lr=2e-5, batch=64, 4xGPU, 无数据增强 |

## 与检测模型的关键差异

|           |         检测         |                   分割                    |
| --------- | :------------------: | :---------------------------------------: |
| end2end   | False (one2many+NMS) |             **True** (无 NMS)             |
| 输出      |    boxes + scores    | boxes + scores + mask_coefficient + proto |
| validator |  DetectionValidator  |             SegmentValidator              |
| per-scale |  concat_flag=False   |           继承 Detect 默认 True           |

## 源码改动

| 文件                                | 改动                             | 原因                                                            |
| ----------------------------------- | -------------------------------- | --------------------------------------------------------------- |
| `head.py:Segment.forward_head()`    | 增加 `concat_flag` 参数          | `Detect.forward()` 传入 `concat_flag=False`，分割头原签名不兼容 |
| `loss.py:v8SegmentationLoss.loss()` | 增加 `teacher_preds`、`**kwargs` | `E2ELoss.__call__()` 统一传 `teacher_preds` 给子 loss           |

## 实验

| 实验 | 配置                           | epochs | lr0  |     box best     |    mask best     |   状态    |
| ---- | ------------------------------ | :----: | :--: | :--------------: | :--------------: | :-------: |
| exp1 | W8A8 + S8 matmul, end2end=True |   50   | 2e-5 | **38.01** (ep31) | **32.66** (ep31) | ✅ 已完成 |

### exp1 精度追踪

|   epoch   |  box mAP  | mask mAP  |
| :-------: | :-------: | :-------: |
|     1     |   37.39   |   31.95   |
| 31 (best) | **38.01** | **32.66** |
|    40     |   37.80   |   32.43   |
| 50 (last) |   37.77   |   32.49   |

### vs float 损耗

| 指标 | float | QAT best | 损耗  |
| ---- | :---: | :------: | :---: |
| box  | 39.6  |  38.01   | -1.59 |
| mask | 33.9  |  32.66   | -1.24 |

分割损耗大于检测（检测 exp33 损耗仅 -1.0 box），分割头多出 mask_coefficient（cv4）和 proto 模块，量化敏感度更高。

### ONNX 导出

```
文件: yolo26_onnx/qat_seg_exp1_one2one_slim.onnx (3.1 MB)
config: 自动检测 config_matmul_s8.json (S8 matmul)
requant: 4 markers (zp_diff + scale_ratio 1.01-1.02)
DQ-Q merge: 1→0
输出: boxes×3 + scores×3 + mask_coefficient + feats×3 + proto×2
```

## 下一步

- [ ] 分割 eval.py 支持（当前仅支持 DetectionValidator）
- [ ] ONNX 精度与训练内验证对比
- [ ] 分析 mask head (cv4) + proto 量化损耗

## 关键文件

- `train-seg.py`: 训练入口
- `ultralytics/cfg/models/26/yolo26-seg.yaml`: 模型结构
- `weights/yolo26n-seg.pt`: 预训练权重
- `ultralytics/cfg/datasets/coco-seg.yaml`: 数据集配置
- `ultralytics/nn/modules/head.py:Segment`: 分割头（继承 Detect）
