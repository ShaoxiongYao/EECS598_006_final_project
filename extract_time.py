import numpy as np

f_list = ["/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_05_12_22_41_59_horizon_5_solver_RL_RRT_cpu_True"]

for filename in f_list:
    length = []
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == "time":
            length.append(float(x[-1]))
    length_np = np.array(length)
    print(length_np.mean())