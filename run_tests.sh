#!/bin/bash

# seed  iterations
# 9     361 
# 13    3389
# 14    276
# 16    275
# 20    205

# Yixuan
# LOG_DIR=/home/yixuan/sshape_boxes_global_1024/seed0
# Shaoxiong laptop
# LOG_DIR=/home/yaosx/Desktop/EECS598_006_final_project/sshape_boxes_global_1024/seed0
# Ensemble on temp_eval_ensemble branch
# LOG_DIR=/home/yaosx/Desktop/EECS598_006_final_project/ensemble_models
LOG_DIR=/home/yixuan/EECS598_006_final_project/ensemble_models

SOLVER_TYPE=RL_RRT
# ENV_NAME=SShape-Boxes-64Pts-Rays-v0
ENV_NAME=SShape-Boxes-1024Pts-SurfaceNormals-v0
# EXP_NAMES=itr_900.pkl,itr_920.pkl,itr_940.pkl,itr_960.pkl,itr_980.pkl
EXP_NAMES=itr_900.pkl

# for HORIZON in $(seq 10 5 81)
for HORIZON in 5
do
FILENAME=test_output/it_comp/$(whoami)_$(date +%y_%m_%d_%H_%M_%S)_horizon_${HORIZON}_solver_${SOLVER_TYPE}_env_${ENV_NAME}_cpu_True
touch $FILENAME
for SEED in {1..100}
# for SEED in 55 82 88 100 107 141 147 172 193 197 211 217 234 248 275 283 285 320 360 378 386 391 393 404 426 458 464 466 486 493 577 597 630 633 653 668 683 685 688 703 722 728 735 753 776 786 824 840 853 857 861 866 890 926 930 934 938 953 967 979 983 998
# for SEED in 4 21 27 36 42 58 61 68 72 76 77 86 91 100 105 107 114 127 129 141 145 147 152 178 179 188 192 196 204 206 220 223 240 263 266 271 284 291 293 298 301 306 322 326 345 349 358 363 367 375 393 400 429 444 448 458 459 464 466 477 482 489 499 504 505 507 514 517 523 554 562 569 576 581 597 603 606 609 620 622 626 669 670 672 678 691 723 728 731 736 738 739 744 747 748 751 753 768 769 777 781 783 795 798 803 804 816 841 857 861 864 866 871 873 875 891 898 902 920 924 928 959 965 970 974 980 # YCB
# for SEED in 1 4 9 21 34 52 56 86 98 104 120 121 145 148 187 189 190 193 199 207 220 264 267 282 283 288 311 326 327 339 341 354 355 428 429 432 433 457 468 481 529 545 553 565 581 582 583 593 603 605 618 619 631 638 651 657 662 665 671 674 679 690 701 709 715 716 720 728 730 731 740 751 755 757 761 774 779 813 816 818 827 849 851 953 965 974 975 # global
do
    python -m nmp.test_run --cpu --obstacles_type handcraft:set_1 --exp-names $EXP_NAMES --seed $SEED --horizon $HORIZON --solver_type $SOLVER_TYPE $ENV_NAME >> $FILENAME
done
done
