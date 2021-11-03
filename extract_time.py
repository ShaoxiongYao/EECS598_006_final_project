import numpy as np

# f_list = ["/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_05_24_02_19_00_horizon_5_solver_RL_RRT_env_SShape-Boxes-256Pts-SurfaceNormals-v0_cpu_True",
# "/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_05_24_02_26_01_horizon_5_solver_Normal_RRT_env_SShape-Boxes-256Pts-SurfaceNormals-v0_cpu_True"]

f_list = [ "/home/yaosx/Desktop/EECS598_006_final_project/test_output/early_stop_trick/yaosx_21_08_22_16_13_57_horizon_20_solver_single_RL_RRT_env_SShape-Boxes-1024Pts-SurfaceNormals-v0_obs_handcraft:set_1",
    "/home/yaosx/Desktop/EECS598_006_final_project/test_output/early_stop_trick/yaosx_21_08_22_16_39_48_horizon_20_solver_single_RL_RRT_env_SShape-Boxes-1024Pts-SurfaceNormals-v0_obs_handcraft:set_1",
    "/home/yaosx/Desktop/EECS598_006_final_project/test_output/early_stop_trick/yaosx_21_08_22_16_48_51_horizon_5_solver_single_RL_RRT_env_SShape-Boxes-1024Pts-SurfaceNormals-v0_obs_handcraft:set_1" ]

key = "time"
# key = "time"

for filename in f_list:
    length = []
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == key:
            length.append(float(x[-1]))
    print(length)
    length_np = np.array(length)
    print(length_np.mean())