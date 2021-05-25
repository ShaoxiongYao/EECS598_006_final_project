import numpy as np

f_list = ["/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_05_24_02_19_00_horizon_5_solver_RL_RRT_env_SShape-Boxes-256Pts-SurfaceNormals-v0_cpu_True",
"/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_05_24_02_26_01_horizon_5_solver_Normal_RRT_env_SShape-Boxes-256Pts-SurfaceNormals-v0_cpu_True"]

for filename in f_list:
    length = []
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == "length:":
            length.append(float(x[-1]))
    length_np = np.array(length)
    print(length_np.mean())