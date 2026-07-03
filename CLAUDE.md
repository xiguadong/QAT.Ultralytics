# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 协作说明

- **Python 环境**：`/home/heqi/miniforge3/envs/torch2.6-qat-yolo`
- **工作重点**：调试 QAT（量化感知训练）模型，跑通 yolo26 检测和分割模型的 QAT 训练
- **目标精度**：yolo26n float=40.0 → qat ≥ 39.2；yolo26s float=47.8 → qat ≥ 47.3
- **回复语言**：中文

## Commands

```bash
# 安装（开发模式）
pip install -e .

# 运行快速测试（排除慢测试）
pytest tests/

# 运行所有测试（含 QAT 训练测试）
pytest tests/ --slow

# 运行单个测试
pytest tests/test_qat_engine.py::test_detect_qat_validate_smoke --slow

# QAT 训练（CLI）
yolo detect train data=coco8.yaml model=yolo26n.yaml epochs=100 qat=True qat_validate=True

# 普通训练
yolo detect train data=coco.yaml model=yolo26n.yaml epochs=100 imgsz=640

# 验证
yolo val detect data=coco.yaml model=yolo26n.pt

# 导出 ONNX
yolo export model=yolo26n.pt format=onnx
```

## 依赖版本约束

`requirements.txt` 中关键版本锁定（与 `pyproject.toml` 中的宽松约束不同）：
- `torch==2.6.0`（PT2E QAT 必须）
- `onnx<1.20.0`
- `onnx-ir<0.1.16`
- `onnxscript==0.4.0`
- `onnxruntime==1.21.0`

## 架构概览

### 核心模块

| 路径 | 作用 |
|------|------|
| `ultralytics/engine/model.py` | YOLO 入口，`train/val/predict/export` 的统一接口 |
| `ultralytics/engine/trainer.py` | 训练循环，含 QAT 的 `qat_model`/`export_model` 状态管理 |
| `ultralytics/engine/validator.py` | 验证，支持 float/exported/prepared 三种模型分支 |
| `ultralytics/engine/exporter.py` | 多格式导出（ONNX、TF、CoreML、TRT 等） |
| `ultralytics/cfg/__init__.py` | CLI 入口点，任务/模式定义 |
| `ultralytics/nn/modules/` | 网络组件（conv、block、head、transformer） |

### QAT 专属模块（本仓库新增）

| 路径 | 作用 |
|------|------|
| `ultralytics/utils/pt2e_bn_patch.py` | Monkey-patch torch，保留 BN 超参（momentum=0.03, eps=1e-3） |
| `ultralytics/utils/qat_utils.py` | PT2E 准备流程：`prepare_pt2e_qat_model()` |
| `ultralytics/utils/ax_quantizer.py` | 自定义量化器 `AXQuantizer`，继承 PT2E quantizer |
| `ultralytics/utils/ax_quantizer_lsq.py` | LSQ 量化器变体 |
| `ultralytics/utils/ax_quantizer_utils.py` | 量化工具函数 |

### QAT 训练数据流

```
model.train(qat=True, qat_validate=True)
  → Trainer.__init__()        # 创建 float_model, export_model, qat_model
  → trainer.train()           # 用 qat_model 替换 float_model 参与梯度更新
      → prepare_pt2e_qat_model()  # torch.export.export_for_training + prepare_qat_pt2e
      → 每 epoch 切换 observer/fake-quant
  → validator.validate()      # 支持 qat_validate=True 时评估 prepared 模型
```

### 模型结构（YOLO26）

- **Backbone**：DarkNet with C2f blocks
- **Neck**：PAFPN（Path Aggregation FPN）
- **Head**：任务相关（Detect / Segment / Classify / Pose / OBB）
- 模型 YAML 配置位于 `ultralytics/cfg/models/`

## 测试结构

```
tests/
├── conftest.py              # pytest hooks，--slow 标志，session cleanup
├── test_qat_engine.py       # QAT 训练烟雾测试（标记 @pytest.mark.slow）
├── test_pt2e_bn_patch.py    # BatchNorm patch 验证
├── test_engine.py           # 训练/验证/导出集成测试
├── test_python.py           # 基础 API 测试
└── test_exports.py          # 各格式导出测试
```

QAT 测试默认被 `--slow` 标志排除，需显式传入才运行。

## 代码质量工具

- **ruff**：格式检查，行宽 120
- **yapf**：PEP8，列宽 120
- **isort**：单行 import
- **codespell**：拼写检查
- pytest 配置：`--doctest-modules --durations=30 --color=yes`
