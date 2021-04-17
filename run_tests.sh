#!/bin/bash

# seed  iterations
# 9     361 
# 13    3389
# 14    276
# 16    275
# 20    205

HORIZON=80
# LOG_DIR=/home/yixuan/sshape_boxes/seed0
LOG_DIR=log_dir

# NMP Failure cases
# for SEED in 55 82 88 100 
#             107 141 147 172 193 197 
#             211 217 234 248 275 283 285 
#             320 360 378 386 391 393
#             404 426 458 464 466 486 493  

# for SEED in {401..500}
for SEED in 493
do
    python -m nmp.test_run --cpu SShape-Boxes-64Pts-Rays-v0 --exp-name $LOG_DIR/params.pkl --seed $SEED --horizon $HORIZON --episodes 0 
done