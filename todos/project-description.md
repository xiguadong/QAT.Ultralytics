# Project: Ultralytics YOLO26 QAT 调试仓

基于 Ultralytics YOLO26 主干接入 PT2E QAT，目标是优先跑通检测与分割模型的量化感知训练。

## Features

- 基于 `torch.export.export_for_training()` 与 `prepare_qat_pt2e()` 的 PT2E QAT 图准备
- 在 `YOLO.train()` / `trainer` / `validator` 链路中支持检测与分割 QAT 训练和在线验证
- 集成自定义 `AXQuantizer`、BN 补丁以及最小化 smoke test

## Tech Stack

- Language: Python
- Framework: PyTorch 2.6, Ultralytics YOLO26
- Build/Env: 当前可用环境为 `/home/heqi/miniforge3/envs/torch2.6-qat-yolo`
- Dependency: torch.ao PT2E QAT, 自定义 quantizer, COCO/YOLO 数据集配置

## Structure

- `ultralytics/`: 训练、验证、模型与 QAT 工具实现
- `tests/`: PT2E BN patch 与 QAT 训练烟测
- `scripts/`: 本地源码启动脚本，如 `train_qat_detect.py`
- `docs/`: QAT 调试记录与阶段计划
- `todos/`: 当前任务分析、执行记录与归档

## Entry Points

- 主入口: `PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python scripts/train_qat_detect.py ...`
- 兼容入口: `PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python - <<'PY' ... PY`

## Commands

- Build: `/home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -m py_compile scripts/train_qat_detect.py`
- Test: `PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -m pytest tests/test_qat_engine.py -k detect`
- Lint/Format: 仓库内暂未发现固定格式化入口，按最小改动维护
- Run: `PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python scripts/train_qat_detect.py --model /home/heqi/project-qat/ultralytics/yolo26n.pt --data ultralytics/cfg/datasets/coco.yaml --device 1 --exist-ok`

## Testing Strategy

- 单元测试位置: `tests/test_pt2e_bn_patch.py`
- 集成测试位置: `tests/test_qat_engine.py`
- 新增测试约定: 优先补 detect / segment 的最小 QAT 训练与验证烟测
