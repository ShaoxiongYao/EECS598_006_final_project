import numpy as np
import matplotlib.pyplot as plt

avg=[]

filename = ["/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_14_41_50_horizon_5_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_14_47_12_horizon_10_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_14_51_54_horizon_15_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_14_56_06_horizon_20_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_15_01_27_horizon_25_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_15_06_14_horizon_30_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_15_10_59_horizon_35_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_15_16_27_horizon_40_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_15_21_59_horizon_45_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_15_28_08_horizon_50_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_15_33_45_horizon_55_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_15_40_04_horizon_60_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_15_46_47_horizon_65_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_15_53_03_horizon_70_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_15_59_59_horizon_75_solver_RL_RRT_cpu_True",
            "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_26_16_06_58_horizon_80_solver_RL_RRT_cpu_True"]
for fn in filename:
    time = []
    f = open(fn, "r")
    for line in f:
        x=line.split()
        if x[0] == "time":
            time.append(float(x[-1]))
    time_np = np.array(time)
    avg.append(time_np.mean())
    f.close()

h=np.arange(5,81,5)
t=np.array(avg)

plt.plot(h,t)
plt.xlabel("horizon")
plt.ylabel("time, s")
plt.title("Horizon vs. time")
plt.savefig("exp2_horizon_time.png")