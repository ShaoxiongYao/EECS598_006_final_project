import numpy as np

f_list = ["/home/yixuan/EECS598_006_final_project/test_output/YCB/yixuan_21_04_29_13_55_36_horizon_5_solver_Normal_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/YCB/yixuan_21_04_29_14_06_56_horizon_5_solver_RL_RRT_cpu_True"]

for filename in f_list:
    length = []
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == "time":
            length.append(float(x[-1]))
    length_np = np.array(length)
    print(length_np.mean())