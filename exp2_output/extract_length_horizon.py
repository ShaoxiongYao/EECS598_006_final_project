import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
avg = []
# filename = [f for f in listdir("/home/yixuan/EECS598_006_final_project/test_output/horizon") if isfile(join("/home/yixuan/EECS598_006_final_project/test_output/horizon", f))]
filename = ["/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_14_58_46_horizon_5_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_16_08_01_horizon_10_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_16_12_53_horizon_15_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_16_17_15_horizon_20_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_16_22_40_horizon_25_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_16_27_29_horizon_30_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_16_32_13_horizon_35_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_16_37_41_horizon_40_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_16_43_13_horizon_45_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_16_49_22_horizon_50_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_16_54_59_horizon_55_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_17_01_18_horizon_60_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_17_08_02_horizon_65_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_17_14_18_horizon_70_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_17_21_14_horizon_75_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/horizon/yixuan_21_04_29_17_28_10_horizon_80_solver_RL_RRT_cpu_True"]
for fn in filename:
    time = []
    f = open(fn, "r")
    for line in f:
        x=line.split()
        if x[0] == "length:":
            time.append(float(x[-1]))
    time_np = np.array(time)
    avg.append(time_np.mean())
    f.close()

h=np.arange(5,81,5)
t=np.array(avg)

plt.plot(h,t)
plt.xlabel("horizon")
plt.ylabel("length")
plt.title("Horizon vs. length")
plt.savefig("exp2_horizon_length.png")