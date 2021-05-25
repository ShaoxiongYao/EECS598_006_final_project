import numpy as np

f_list = ["/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_05_24_00_50_15_horizon_80_solver_RL_env_SShape-Boxes-256Pts-SurfaceNormals-v0_cpu_True"]

cur_seed = 0

for filename in f_list:
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == "seed:":
            cur_seed = int(x[-1])
        if x[0] == "successes" and x[-1] == "[array([False])]":
            print(cur_seed, end=' ')
