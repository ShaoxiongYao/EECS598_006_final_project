#!/bin/bash

# seed  iterations
# 9     361 
# 13    3389
# 14    276
# 16    275
# 20    205

HORIZON=80

for SEED in {101..200}
do
    python -m nmp.test_run --cpu SShape-Boxes-64Pts-Rays-v0 --exp-name log_dir/params.pkl --seed $SEED --horizon $HORIZON --episodes 0 
done