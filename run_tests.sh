#!/bin/bash

# seed  iterations
# 9     361 
# 13    3389
# 14    276
# 16    275
# 20    205

HORIZON=8
LOG_DIR=/home/yixuan/sshape_boxes/seed0

#for SEED in {101..200}
for SEED in 55 82 88 100 107 141 147 172 193 197
do
    python -m nmp.test_run --cpu SShape-Boxes-64Pts-Rays-v0 --exp-name $LOG_DIR/params.pkl --seed $SEED --horizon $HORIZON --episodes 0 
done