import numpy as np

f_list = ["/home/yixuan/EECS598_006_final_project/test_output/hand/yixuan_21_04_29_15_19_03_horizon_5_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/hand/yixuan_21_04_29_15_23_15_horizon_5_solver_RL_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/hand/yixuan_21_04_29_15_21_03_horizon_5_solver_Normal_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/hand/yixuan_21_04_29_15_26_35_horizon_5_solver_Normal_RRT_cpu_True_free",
            "/home/yixuan/EECS598_006_final_project/test_output/hand/yixuan_21_04_29_15_27_26_horizon_5_solver_Normal_RRT_cpu_True_bridge",
            "/home/yixuan/EECS598_006_final_project/test_output/hand/yixuan_21_04_29_15_27_54_horizon_5_solver_Normal_RRT_cpu_True_nearsurface"]

for filename in f_list:
    length = []
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == "length:":
            length.append(float(x[-1]))
    length_np = np.array(length)
    print(length_np.mean())