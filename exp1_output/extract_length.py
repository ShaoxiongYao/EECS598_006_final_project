import numpy as np

f_list = ["/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_29_14_38_14_horizon_5_solver_Normal_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_29_14_31_40_horizon_5_solver_RL_RRT_cpu_True"]

for filename in f_list:
    length = []
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == "length:":
            length.append(float(x[-1]))
    length_np = np.array(length)
    print(length_np.mean())