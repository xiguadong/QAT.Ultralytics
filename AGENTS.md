# 仓库协作说明

- Python 环境应当使用：`miniforge3/envs/torch2.6-qat-yolo`。
- 当前仓库用于调试 QAT（量化感知训练）模型。
- 所有回复统一使用中文。
- 优先解决qat训练bug，跑通yolo26检测模型和分割模型的qat训练
- yolo26n浮点模型精度40.2（end2end=True），qat模型目标精度大于等于39.2；yolo26s浮点模型精度为47.8（end2end=True），qat模型目标精度大于47.3。
- yolo26n浮点模型精度40.9（end2end=False），qat模型目标精度大于等于39.9；yolo26s浮点模型精度为48.6（end2end=False），qat模型目标精度大于48.0。