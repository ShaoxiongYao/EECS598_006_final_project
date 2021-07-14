#!/bin/bash
# training script

LOG_DIR=sshape_boxes_global_1024
# ENV_NAME=SShape-Boxes-64Pts-Rays-v0
ENV_NAME=SShape-Boxes-1024Pts-SurfaceNormals-v0

for SEED in 1
do
python -m nmp.train $ENV_NAME $LOG_DIR --horizon 80 --seed $SEED
done
