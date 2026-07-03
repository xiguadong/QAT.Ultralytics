# Analysis - [Exp-4] KD-QAT 检测模型实现

## 1. 需求拆解

### 背景

Exp-1 与 Exp-2 已经基本排除“仅靠 head 跳过量化”或“仅靠训练日程调整”即可把 yolo26n 全 INT8 QAT 拉回目标精度的可能性。当前 `exp2-int8` 跑满 100 epoch 后最佳 `mAP50-95` 仍只有 `37.617`，显著低于目标 `39.2`，因此需要进入更强的训练策略方案。

### 目标

在现有 PT2E QAT 检测训练链路中，为 QAT student 增加 float teacher 蒸馏监督，形成可启动、可记录、可继续调参的 KD-QAT 基线实现。

### 输入/输出

- 输入：`todos/quant_plan.md`、现有 QAT prepare/resume 修复、`weights/yolo26n.pt`、当前 detect trainer/loss 链路
- 输出：KD-QAT 配置项、teacher 挂接实现、检测 loss 中的 KD loss、最小可运行验证结果

### 成功标准

- [ ] `qat_kd=True` 时 QAT 检测训练可正常进入 epoch
- [ ] student 总 loss 中包含 KD 项，训练日志能看到对应指标
- [ ] `qat_kd=False` 时原有 detect / QAT 路径行为不变

## 2. 代码定位

### 相关目录

- `ultralytics/engine`
- `ultralytics/nn`
- `ultralytics/models/yolo/detect`
- `ultralytics/utils`
- `ultralytics/cfg`

### 关键文件

- `ultralytics/engine/model.py`: `YOLO.train()` 中 QAT prepare，当前已构造 `float_model` 并导出 training ONNX
- `ultralytics/engine/trainer.py`: 训练主循环，在 QAT 模式下使用 `self.qat_model(batch["img"])` + `unwrap_model(self.model).loss(batch, preds)`
- `ultralytics/nn/tasks.py`: `DetectionModel.loss()` 与 `init_criterion()` 的入口
- `ultralytics/utils/loss.py`: `v8DetectionLoss` 是 detect QAT 当前主 loss 实现
- `ultralytics/cfg/default.yaml`: QAT 配置项注册位置

### 现有模式

- QAT 准备阶段已经在 `model.py:_prepare_qat_training()` 构造出 `float_model = deepcopy(source_model).float().train()`
- 训练循环中 loss 仍通过 eager `DetectionModel.loss()` 调用 criterion，因此可以把 KD loss 收敛到 detection criterion，而不必重写整个训练循环
- `DetectionTrainer.loss_names` 当前固定为 `("box_loss", "cls_loss", "dfl_loss")`，若增加 KD 项，需要同步扩展日志键

## 3. 方案设计

### 方案 A

- 做法：在 trainer 上持有 `teacher_model`，训练循环把 `teacher_preds` 一起传给 detection criterion，由 criterion 统一计算 task loss + KD loss
- 优点：teacher 生命周期清晰，KD 逻辑集中在 loss 层，便于以后扩展到 segment
- 风险：需要调整训练循环与 `DetectionModel.loss()` 的入参形态

### 方案 B

- 做法：在 `v8DetectionLoss` 内部直接从 model/trainer 反查 teacher，再自行做前向
- 优点：表面改动点少
- 风险：loss 层耦合 trainer 状态，接口不清晰，也不利于后续调试

### 选型结论

采用方案 A。先在 trainer 明确维护 `teacher_model`，训练循环在 QAT+KD 模式下额外跑一次 `teacher_model(batch["img"])`，再把 `student_preds` 和 `teacher_preds` 一起送入 detection criterion。这样可以最小化副作用，并保持 `qat_kd=False` 时完全旁路。

## 4. 实施计划映射

- 步骤 1 -> 文件：`ultralytics/cfg/default.yaml`
- 步骤 2 -> 文件：`ultralytics/engine/model.py`、`ultralytics/engine/trainer.py`
- 步骤 3 -> 文件：`ultralytics/nn/tasks.py`、`ultralytics/utils/loss.py`、`ultralytics/models/yolo/detect/train.py`
- 步骤 4 -> 验证：detect QAT+KD 最小训练命令

## 5. 测试策略

### 自动化测试

- 命令：优先运行最小化 detect QAT+KD 训练烟测，必要时补 `pytest`
- 预期：训练可进入 epoch，日志中新增 KD loss 项，`qat_kd=False` 不回归
- 当前结果：
- `PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -m py_compile ultralytics/engine/model.py ultralytics/engine/trainer.py ultralytics/nn/tasks.py ultralytics/models/yolo/detect/train.py ultralytics/utils/loss.py` 通过
- `coco8.yaml` + `qat=True` + `qat_kd=True` + `qat_validate=False` 烟测已跑通，训练日志新增 `kd_loss`
- `coco8.yaml` + `qat=True` + `qat_kd=True` + `qat_validate=True` 烟测已跑通，训练与 validator 全链路正常结束
- 正式实验 `runs/detect/exp4-kd-int8` 已跑满 `50 epoch`
- 最佳指标：`epoch 45 -> metrics/mAP50(B)=53.207, metrics/mAP50-95(B)=37.769`
- 最终指标：`epoch 50 -> metrics/mAP50(B)=53.312, metrics/mAP50-95(B)=37.610`
- 续跑过程中额外修复 1 个 resume bug：若不保留 `qat_kd*` 参数，恢复训练后 `teacher_model` 不会重建，`kd_loss` 会错误变成 `0`

### 手动验证

- 步骤：检查训练启动日志、teacher 是否被冻结、loss 键是否包含 KD 项
- 预期：teacher 不参与梯度更新，student loss 中出现 KD 指标
- 当前结论：teacher 准备、KD loss 日志、resume 后 KD 恢复均已验证通过

### 回归范围

- QAT 检测训练主循环
- 非 QAT 检测训练路径
- QAT resume 链路
- 验证器对 loss 项数量的兼容性

## 6. 风险与回滚

- 风险 1 -> 对策：teacher 输出结构与 student 不完全一致，先以 detect 输出字典对齐后再计算 KD
- 风险 2 -> 对策：loss 项数量变化影响日志/validator，需同步扩展 `loss_names` 与 `label_loss_items`
- 风险 3 -> 对策：teacher 占用额外显存，先做 detect 最小 batch 烟测确认可跑
- 回滚方案：保留配置开关，出现问题时通过 `qat_kd=False` 完全禁用；必要时单独回退 KD 相关文件

## 7. 阶段性结论

- Exp-4 在 `50 epoch` 内证明了 KD-QAT 对全 INT8 路径有小幅增益，但提升幅度有限
- 其最优 `mAP50-95=37.769` 高于 Exp-2 的 `37.617`，但仍低于 Exp-1 的 `38.321`
- 距目标 `39.2` 仍差约 `1.431 mAP`，因此按当前 KD 形式与超参，Exp-4 不能视为达标方案
