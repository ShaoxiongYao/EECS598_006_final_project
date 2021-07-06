import numpy as np

f_list = ["/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_07_06_01_47_28_horizon_5_solver_Normal_RRT_env_SShape-Boxes-1024Pts-SurfaceNormals-v0_cpu_True",
"/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_07_06_02_08_55_horizon_5_solver_RL_RRT_env_SShape-Boxes-1024Pts-SurfaceNormals-v0_cpu_True",
"/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_07_06_02_27_51_horizon_5_solver_RL_RRT_env_SShape-Boxes-1024Pts-SurfaceNormals-v0_cpu_True"]

for filename in f_list:
    length = []
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == "time":
            length.append(float(x[-1]))
    length_np = np.array(length)
    print(length_np.mean())