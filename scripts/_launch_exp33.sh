#!/bin/bash
cd /home/heqi/project-qat/ultralytics
exec /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u train.py >> /tmp/exp33.log 2>&1
