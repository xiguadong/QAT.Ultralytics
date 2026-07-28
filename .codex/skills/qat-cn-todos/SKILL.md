---
name: qat-cn-todos
description: 中文 QAT 任务落地协议。用于在 QAT.Ultralytics 仓库中把 todos/todos.md 条目转化为可提交实现，并维护 task.md 与 analysis.md。遇到需要中文连续推进、最少确认、QAT 调试验证与归档提交的任务时使用。
---

# QAT CN Todos

## 目标

- 将 todo 条目快速转为可提交实现。
- 保持中文沟通、连续推进、最少确认。
- 适配当前仓库：QAT 模型调试；运行前激活项目兼容的 Python/QAT 环境。

## 能力选择

- `plan`：需求边界不清、改动跨模块或风险较高时先拆计划。
- `implement`：按计划执行改动并完成验证。
- `debug_quant_error`：量化导出、推理或校验失败时专项排查。

## 工作流

### 1. INIT

1. 读取 `todos/project-description.md`。
2. 若不存在，基于 `templates/project-description.md` 创建。
3. 确保目录存在：`todos/work`、`todos/done`。
4. 在执行 Python 命令前确认已激活项目兼容环境，并使用当前环境的 `python`。

### 2. SELECT

1. 读取 `todos/todos.md`。
2. 默认选择最高优先级条目；仅在优先级冲突或信息不足时询问用户。
3. 创建任务目录：
   - `TIMESTAMP=$(date +%Y%m%d-%H%M%S)`
   - `TASK_DIR="todos/work/${TIMESTAMP}-<task-title-slug>"`
4. 使用模板初始化：
   - `cp .codex/skills/qat-cn-todos/templates/task.md "${TASK_DIR}/task.md"`
   - `cp .codex/skills/qat-cn-todos/templates/analysis.md "${TASK_DIR}/analysis.md"`
5. 从 `todos/todos.md` 删除已选条目。

### 3. REFINE

1. 调研代码路径、现有实现与风险，补充到 `analysis.md`。
2. 在 `task.md` 中完善 Description、Implementation Plan、Validation。
3. 用户新增约束时，先更新 `task.md` 与 `analysis.md`，再继续实现。
4. 将任务状态改为 `InProgress`。

### 4. IMPLEMENT

1. 按计划复选框逐项完成，完成即勾选。
2. 每轮改动后执行必要验证（lint/test/run/export）并记录结果。
3. 出现量化相关失败时调用 `debug_quant_error`，并补充根因和修复记录。
4. 完成后将状态改为 `AwaitingCommit`。

### 5. COMMIT

1. 汇总改动、验证命令与结果。
2. 生成提交信息：`[任务标题]: [变更摘要]`。
3. 单次确认后执行提交。
4. 归档到 `todos/done/<task-dir>/` 并保留 `task.md`、`analysis.md`。`todos/` 是本地记录，不随交付代码提交。

## 模板文件

- `templates/task.md`
- `templates/analysis.md`
- `templates/project-description.md`

## 输出规范

- 全程中文回复。
- 先给结论，再说明改动和文件路径。
- 默认自动推进，仅在必要节点确认。
