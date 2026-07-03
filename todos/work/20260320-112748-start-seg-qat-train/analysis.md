# Analysis - 开始分割模型 QAT 训练

## 1. 需求拆解

### 背景

检测模型 PT2E QAT 训练和导出链路已经修通，用户要求继续推进分割模型，并明确指定使用 `train-seg.py` 与 `coco8-seg.yaml`。在训练跑通后，用户进一步要求扩展现有 `export.py`，让同一个脚本也能导出分割 QAT ONNX。

### 目标

跑通 `train-seg.py` 对 `yolo26n-seg` 的 QAT 训练，修复分割训练链路中的数据配置阻塞，并扩展 `export.py` 支持分割 QAT ONNX 导出。

### 输入/输出

- 输入：`train-seg.py`、`ultralytics/cfg/datasets/coco8-seg.yaml`、`weights/yolo26n-seg.pt`、当前已修复的 PT2E QAT 公共代码、现有 `export.py`
- 输出：可运行的分割 QAT 训练命令、必要修复、可用的分割 QAT 导出脚本、10 epoch 运行结果与任务记录

### 成功标准

- [x] `train-seg.py` 可在 `coco8-seg.yaml` 上启动分割 QAT 训练
- [x] 训练前 float dynamo ONNX 导出成功
- [x] 分割 QAT 训练和验证跑完 10 epoch
- [x] `export.py --task segment` 成功导出可被 ORT 加载的分割 QAT ONNX

## 2. 代码定位

### 相关目录

- `ultralytics/engine`
- `ultralytics/nn/modules`
- `ultralytics/utils`
- `ultralytics/cfg/datasets`

### 关键文件

- `train-seg.py`: 用户指定的分割 QAT 启动入口
- `ultralytics/cfg/datasets/coco8-seg.yaml`: 分割数据集路径与类别配置
- `ultralytics/engine/model.py`: 训练前 float ONNX 导出、QAT prepare 入口
- `ultralytics/utils/tal.py`: 公共 anchor 生成逻辑，已修过 detect/segment 共用的 `aten.item` 导出问题
- `export.py`: 现有 QAT 导出脚本，本轮扩展为 detect/segment 双任务共用入口

### 现有模式

分割模型复用和检测模型一致的 PT2E QAT 准备流程：先在 `YOLO.train()` 中做训练前 float ONNX debug 导出，再通过 `prepare_pt2e_qat_model()` 挂接 QAT graph，随后进入 trainer 正常训练和 validator 验证。由于 `Segment26` 继承了检测头的一部分推理逻辑，检测路径上修复的 dynamo 导出问题会直接影响分割模型。

补充确认：

- `Segment26` 的 QAT 模型在 `eval` 下返回顶层 `dict`，包含 `one2many` 和 `one2one`
- `one2one` 中包含 `boxes`、`scores`、`mask_coefficient`、`feats`、`proto`
- 当前 `proto` 为长度 2 的 tuple，形状分别是 `(1, 32, 160, 160)` 和 `(1, 80, 80, 80)`，可命名为 `proto_masks`、`proto_semseg`

## 3. 方案设计

### 方案 A

- 做法：直接执行 `train-seg.py`，按真实日志逐步修复首个阻塞点；训练跑通后，再扩展 `export.py` 复用同一条量化构图路径
- 优点：最快暴露分割 QAT 真正的问题，避免只做静态阅读
- 风险：会自动下载数据和权重，需要额外确认路径是否一致

### 方案 B

- 做法：先静态检查 `Segment26`、seg loss、validator，再决定是否运行
- 优点：上下文更完整
- 风险：推进慢，而且很多 PT2E/QAT 问题只能在真实导出和训练时出现

### 选型结论

采用方案 A。直接运行 `train-seg.py`，以真实训练链路为主线修复阻塞；在此基础上扩展 `export.py`，保留 detect 默认行为并增加 `--task segment`。

## 4. 实施计划映射

- 步骤 1 -> 文件：`train-seg.py`、`ultralytics/cfg/datasets/coco8-seg.yaml`
- 步骤 2 -> 文件：`ultralytics/engine/model.py`、`ultralytics/utils/tal.py`
- 步骤 3 -> 文件：`export.py`
- 步骤 4 -> 文件：运行日志、`runs/segment/qat2`

## 5. 测试策略

### 自动化测试

- 命令：未新增自动化测试，直接用用户指定脚本和导出脚本做端到端验证
- 预期：分割模型训练前导出成功，QAT prepare 成功，训练和验证正常推进；分割 QAT ONNX 导出后可通过 checker 和 ORT

### 手动验证

- 步骤：运行 `env PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python train-seg.py`
- 预期：日志出现 `Exported dynamo float ONNX to dynamo_float.onnx.`、`Prepared PT2E QAT model...`、每 epoch 的 box/seg/cls/dfl/sem loss 与 mask mAP 输出，最终打印 `10 epochs completed`
- 步骤：运行 `env PYTHONPATH=/home/heqi/project-qat/ultralytics /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python export.py --task segment`
- 预期：生成 `qat-seg.onnx` 与 `qat-seg_slim.onnx`，输出为 `boxes/scores/mask_coefficient/feat_p3/feat_p4/feat_p5/proto_masks/proto_semseg`，并通过 checker 与 ORT

### 回归范围

分割训练入口、训练前 ONNX 导出、公共 anchor 生成逻辑、seg 数据集路径解析、QAT validator、QAT ONNX 导出脚本

### 实际结果

- 初次运行阻塞在数据集路径：`coco8-seg.yaml` 的 `path` 指向 `/home/heqi/dataset/coco8-seg`，而实际自动下载目录是 `/home/heqi/project/datasets/coco8-seg`
- 修复后再次运行：分割模型训练前 float dynamo ONNX 导出成功，未回退 legacy exporter
- QAT prepare 成功：日志出现 `Prepared PT2E QAT model with fixed spatial size=640 and dynamic batch<= 128.`
- 训练与验证成功：完整跑完 10 epoch，最终日志为 `10 epochs completed in 0.041 hours.` 和 `Skipping checkpoint-based final_eval for PT2E QAT model.`
- 导出脚本扩展完成：`export.py` 现支持 `--task detect|segment`，detect 默认行为保持不变
- 分割导出成功：`export.py --task segment` 成功生成 `qat-seg.onnx` 与 `qat-seg_slim.onnx`
- 导出产物校验成功：`qat-seg.onnx`、`qat-seg_slim.onnx` 均通过 `onnx.checker` 与 `onnxruntime`，输出为 `boxes`、`scores`、`mask_coefficient`、`feat_p3`、`feat_p4`、`feat_p5`、`proto_masks`、`proto_semseg`

## 6. 风险与回滚

- 风险 1 -> 对策：当前 `coco8-seg.yaml` 使用绝对路径，若后续仓库迁移机器需重新调整数据根目录
- 风险 2 -> 对策：当前只验证了 `coco8-seg` 小数据集；更大分割数据集仍可能触发显存或导出性能问题
- 风险 3 -> 对策：当前分割导出包装器基于 `Segment26` 的 `proto=(proto_masks, proto_semseg)` 结构命名；若后续更换为别的分割头，脚本会自动回退到 `proto_0/proto_1/...` 命名
- 回滚方案：若需要恢复原始数据配置，只回退 `ultralytics/cfg/datasets/coco8-seg.yaml` 的 `path`；若要回退导出扩展，仅回退 `export.py` 即可，detect/segment 训练链路修复与其解耦
