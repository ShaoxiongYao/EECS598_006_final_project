import numpy as np

f_list = ["/home/yixuan/EECS598_006_final_project/test_output/normal/yixuan_21_04_29_14_50_09_horizon_5_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/normal/yixuan_21_04_29_14_43_02_horizon_5_solver_Normal_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/normal/yixuan_21_04_29_09_16_49_horizon_80_solver_RL_cpu_True"]

for filename in f_list:
    length = []
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == "length:":
            if filename == f_list[-1]:
                length.append(float(x[-1][1:-1]))
            else:
                length.append(float(x[-1]))
    length_np = np.array(length)
    print(length_np.mean())