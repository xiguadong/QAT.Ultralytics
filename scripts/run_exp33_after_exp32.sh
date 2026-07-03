#!/bin/bash
# 定时任务：等 exp32 训练完成后自动启动 exp33
# exp33 = 与 exp32 相同，但 matmul/gridsample 使用 S8（config_matmul_s8.json）
# 使用方式: nohup bash scripts/run_exp33_after_exp32.sh &> exp33_launcher.log &

PROJECT_DIR="/home/heqi/project-qat/ultralytics"
PYTHON="/home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python"
EXP32_DIR="${PROJECT_DIR}/runs/detect/exp32-yolo26n-S16matmul-e2eFalse"
EXP33_NAME="exp33-yolo26n-S8matmul-e2eFalse"

cd "${PROJECT_DIR}"

echo "============================================================"
echo "[$(date)] 开始监控 exp32 训练进度..."
echo "exp32 目录: ${EXP32_DIR}"
echo "exp33 名称: ${EXP33_NAME}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 检测 exp32 是否完成
# ---------------------------------------------------------------------------
POLL_INTERVAL=60 # 秒

while true; do
  # 检查是否有 exp32 相关的 training 进程在运行
  # train.py 进程特征：python ... train.py
  RUNNING_COUNT=$(ps aux | grep -E "python.*train\.py" | grep -v grep | wc -l)

  # 检查 epoch 50 是否已保存（即 epoch49.pt 存在，0-indexed）
  EPOCH49_EXISTS=""
  if [ -f "${EXP32_DIR}/weights/epoch49.pt" ]; then
    EPOCH49_EXISTS="yes"
  fi

  # 检查 results.csv 的行数（表头 + 50 行 = 51 行表示完成）
  CSV_LINES=0
  if [ -f "${EXP32_DIR}/results.csv" ]; then
    CSV_LINES=$(wc -l < "${EXP32_DIR}/results.csv")
  fi

  NOW=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[${NOW}] 进程数=${RUNNING_COUNT}, epoch49.pt=${EPOCH49_EXISTS:-no}, csv行数=${CSV_LINES}"

  # 判断完成条件：没有 train.py 进程在运行 且 (epoch49.pt 存在 或 csv 行数 >= 51)
  if [ "${RUNNING_COUNT}" -eq 0 ]; then
    if [ "${EPOCH49_EXISTS}" = "yes" ] || [ "${CSV_LINES}" -ge 51 ]; then
      echo ""
      echo "============================================================"
      echo "[$(date)] exp32 训练已完成！开始启动 exp33..."
      echo "============================================================"
      break
    else
      echo "[$(date)] 警告：进程已退出但未检测到完成标志，继续等待..."
    fi
  fi

  sleep "${POLL_INTERVAL}"
done

# ---------------------------------------------------------------------------
# 启动 exp33：与 exp32 完全相同的参数，仅 matmul/gridsample 换 S8 配置
# ---------------------------------------------------------------------------

CMD="${PYTHON} -u train.py \
    --model yolo26n.yaml \
    --data coco.yaml \
    --pretrained yolo26n.pt \
    --epochs 50 \
    --batch 64 \
    --imgsz 640 \
    --device 2 \
    --workers 8 \
    --project ${PROJECT_DIR}/runs/detect/ \
    --name ${EXP33_NAME} \
    --exist-ok \
    --optimizer SGD \
    --lr0 2e-5 \
    --lrf 0.1 \
    --momentum 0.937 \
    --weight-decay 0.0005 \
    --warmup-epochs 0.0 \
    --warmup-momentum 0.8 \
    --warmup-bias-lr 0.1 \
    --box 7.5 \
    --cls 0.5 \
    --dfl 1.5 \
    --seed 0 \
    --deterministic \
    --close-mosaic 0 \
    --amp false \
    --fraction 1 \
    --qat true \
    --qat-config config_matmul_s8.json \
    --qat-dynamic-batch-max 128 \
    --qat-validate true \
    --qat-kd false \
    --end2end false \
    --val true \
    --save true \
    --save-period 1 \
    --plots true"

echo ""
echo "[$(date)] 启动命令:"
echo "  ${CMD}"
echo ""
nohup ${CMD} &> "${EXP33_NAME}.log" &
EXP33_PID=$!
echo "[$(date)] exp33 已启动 (PID=${EXP33_PID})，日志: ${EXP33_NAME}.log"
echo "监控: tail -f ${EXP33_NAME}.log"
echo "============================================================"
