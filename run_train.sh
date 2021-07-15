#!/bin/bash
# training script

LOG_DIR=sshape_boxes_global_1024
# ENV_NAME=SShape-Boxes-64Pts-Rays-v0
ENV_NAME=SShape-Boxes-1024Pts-SurfaceNormals-v0
GPU_ID=1

EPOCHS=1000

for SEED in 1
do
python -m nmp.train --snapshot-mode gap --snapshot-gap 20 $ENV_NAME $LOG_DIR --epochs $EPOCHS --gpu_id $GPU_ID --horizon 80 --seed $SEED
done
