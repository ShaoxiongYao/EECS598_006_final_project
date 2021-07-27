import numpy as np

# f_list = ["/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_05_24_02_19_00_horizon_5_solver_RL_RRT_env_SShape-Boxes-256Pts-SurfaceNormals-v0_cpu_True",
# "/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_05_24_02_26_01_horizon_5_solver_Normal_RRT_env_SShape-Boxes-256Pts-SurfaceNormals-v0_cpu_True"]
# f_list = ['test_output/connect_change/yixuan_21_07_06_01_00_58_horizon_5_solver_Normal_RRT_env_SShape-Boxes-1024Pts-SurfaceNormals-v0_cpu_True']
f_list = ['test_output/it_comp/ensemble_boxes', 'test_output/it_comp/single_boxes']
# f_list = ['test_output/it_comp/ensemble_set_1', 'test_output/it_comp/single_set_1']

# key = "iterations:"
key = "time"

for filename in f_list:
    print(filename)
    data_list = []
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == key:
            data_list.append(float(x[-1]))
    data_np = np.array(data_list)
    # data_np = data_np[data_np>10]

    print("mean:", data_np.mean())
    print("std:", data_np.std())