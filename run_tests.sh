#!/bin/bash

# seed  iterations
# 9     361 
# 13    3389
# 14    276
# 16    275
# 20    205

LOG_DIR=/home/yixuan/sshape_boxes/seed0
# LOG_DIR=log_dir
SOLVER_TYPE=RL_RRT

# NMP Failure cases

# for SEED in {866..1000}

for HORIZON in $(seq 20 5 81)
do
FILENAME=test_output/$(whoami)_$(date +%y_%m_%d_%H_%M_%S)_horizon_${HORIZON}_solver_${SOLVER_TYPE}_cpu_True
touch $FILENAME
for SEED in 55 82 88 100 107 141 147 172 193 197 211 217 234 248 275 283 285 320 360 378 386 391 393 404 426 458 464 466 486 493 577 597 630 633 653 668 683 685 688 703 722 728 735 753 776 786 824 840 853 857 861 866 890 926 930 934 938 953 967 979 983 998
# for SEED in 776 786 824 840 853 857 861 866 890 926 930 934 938 953 967 979 983 998
do
    python -m nmp.test_run --cpu SShape-Boxes-64Pts-Rays-v0 --exp-name $LOG_DIR/params.pkl --seed $SEED --horizon $HORIZON --episodes 0 --solver_type $SOLVER_TYPE >> $FILENAME
done
done