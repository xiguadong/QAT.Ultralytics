# Analysis - [Exp-1] 跳过检测头量化敏感性分析

## 1. 需求拆解

### 背景

`todos/quant_plan.md` 将 Exp-1 定义为敏感性分析实验，目标是定量判断 YOLO26 检测头 `model.23` 对量化损耗的贡献。现有 `config.json` 只有 global A8W8 配置，`ax_quantizer.py` 也默认 regional `module_config` 一定非空，无法表达“指定节点保持 FP32”。

### 目标

打通 `regional_configs -> load_regional_config -> conv annotator` 这条链路，使其能够接受 `module_config: null`，并在 PT2E QAT 图里取消目标 conv 的 activation/weight/output qspec。

### 输入/输出

- 输入：`todos/quant_plan.md`、`config.json`、`ultralytics/utils/ax_quantizer.py`、`ultralytics/utils/ax_quantizer_utils.py`、`weights/yolo26n.pt`
- 输出：支持 head FP32 直通的 quantizer 配置能力，以及对应最小验证记录

### 成功标准

- [x] regional `module_config: null` 能被成功加载
- [x] 目标 head conv 节点的 quantization annotation 被清空为 FP32 直通
- [x] 最小验证可以列出被跳过量化的检测头节点

## 2. 代码定位

### 相关目录

- `ultralytics/utils`
- 仓库根目录 `config.json`
- `todos/work/20260409-111102-exp-1-skip-detect-head-quant`

### 关键文件

- `ultralytics/utils/ax_quantizer.py`: quant config 解析与 regional annotator 分发
- `ultralytics/utils/ax_quantizer_utils.py`: conv/linear 等 op 的 PT2E quantization annotation 实现
- `config.json`: 当前全局 A8W8 配置

### 现有模式

当前 global annotator 会先为所有 conv 写入 input/output/weight qspec，regional annotator 只支持“覆写为另一套量化参数”，不支持撤销已有量化 annotation。

## 3. 方案设计

### 方案 A

- 做法：在 `load_regional_config()` 中放行 `module_config=None`，并在 conv regional 分支将目标节点的 input/weight/bias/output qspec 全部改成 `None`
- 优点：改动最小，保持现有 regional 配置结构不变
- 风险：需确认 PT2E 对 `None qspec` 的处理符合“FP32 直通”预期

### 方案 B

- 做法：在 global annotation 前过滤掉检测头节点，regional 只做白名单量化
- 优点：语义更直接
- 风险：需要更大范围重构 annotator 调用顺序，不适合作为 Exp-1 起步实验

### 选型结论

采用方案 A。先最小化改动打通实验能力，再通过最小图验证观察 head 节点是否仍带有 fake quant。

## 4. 实施计划映射

- 步骤 1 -> 文件：`ultralytics/utils/ax_quantizer.py`
- 步骤 2 -> 文件：`ultralytics/utils/ax_quantizer_utils.py`
- 步骤 3 -> 文件：`config.json`
- 步骤 4 -> 验证：导出训练图并检查 `model.23` 对应 conv 节点 annotation

## 5. 测试策略

### 自动化测试

- 命令：基于 `weights/yolo26n.pt` 运行最小 `torch.export.export_for_training + prepare_pt2e_qat_model` 检查脚本
- 预期：检测头配置命中的 conv 节点 input/output/weight qspec 为 `None`

### 手动验证

- 步骤：打印训练图中 `source_fn_stack` 包含 `model.23` 的 conv 节点名，写入 `config.json` 后重新 prepare QAT 图并检查 annotation
- 预期：命中节点保持 FP32，非 head 节点仍维持 global A8W8

### 验证结果

- `torch.export.export_for_training()` 导出的训练图中，`model.23.*` 共定位到 48 个 `aten.conv2d.default` 节点，对应 `conv2d_78` 到 `conv2d_125`
- regional `module_config: null` 生效后，annotated 导出图中的 48 个 head conv 的 `input_qspec_map` 与 `output_qspec` 全部为 `None`
- 为通过 `prepare_qat_pt2e()`，补充清除了 head 末端 conv 下游 `view_*` 节点残留的 `SharedQuantizationSpec` 引用
- `annotate_bias()` 额外跳过了 input/weight 已置空的 conv，避免 bias 被重新补成 `DerivedQuantizationSpec`
- `prepare_pt2e_qat_model()` 成功后，prepared 图里仍可见的 12 个 head 末端 conv（`conv2d_80/83/.../125`）也全部保持 `None qspec`
- `train.py` 已在 `runs/detect/qat3` 成功启动，并进入 `epoch 1/10`，初始速度约 `2.1 it/s`

### 实验结果

- 实际运行目录：`runs/detect/qat3`
- 训练已完成至 `epoch 10`
- 最佳指标：`epoch 9 -> metrics/mAP50(B)=0.53672, metrics/mAP50-95(B)=0.38321`
- 最终指标：`epoch 10 -> metrics/mAP50(B)=0.53463, metrics/mAP50-95(B)=0.38180`
- 相对 `qat2` 最优 `37.53`，Exp-1 最佳提升约 `+0.79 mAP`
- 相对目标 `39.2`，仍差 `0.879 mAP`

### 结论

- 跳过检测头量化确实能带来可测提升，说明 head 存在敏感性，但贡献幅度有限
- 由于最佳结果 `38.321 < 38.5`，更接近原计划中“head 非主因或非唯一主因”的分支
- 因此不能把后续主方向放在“仅通过 head FP32/mixed precision 即可达标”这一假设上
- 后续若继续做训练策略验证，`Exp-2` 应恢复全 INT8 配置，避免和 Exp-1 的 head skip 结果混淆

### 补充排查

- 在完成 `qat3` 续跑的过程中，还修复了 QAT resume 链路：
- `last.pt` 加载后保留 `qat*` overrides，避免 `resume=True` 时丢失 QAT prepare
- resume 时恢复 checkpoint 中的 `qat_model` state 到 prepared PT2E 图
- 重新打开从 checkpoint 加载出的 float 模型参数 `requires_grad`，避免续跑时 loss 无梯度
- 这些修复不改变 Exp-1 的精度结论，但使后续长周期实验可被中断后恢复

### 回归范围

`ax_quantizer.py` regional 配置加载、conv annotator regional override、QAT prepare 入口的 quant config 解析

## 6. 风险与回滚

- 风险 1 -> 对策：若 `None qspec` 不能稳定取消量化，则改为显式清除节点 annotation 并重建下游所需输入 qspec
- 风险 2 -> 对策：FX 节点名随导出图变化，先用脚本实测而不是猜测配置
- 回滚方案：回退 `config.json` 的 `regional_configs` 和 quantizer 处理逻辑，恢复全局 A8W8
