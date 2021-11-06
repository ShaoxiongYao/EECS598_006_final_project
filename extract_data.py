import numpy as np
import matplotlib.pyplot as plt

# f_list = ["/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_05_24_02_19_00_horizon_5_solver_RL_RRT_env_SShape-Boxes-256Pts-SurfaceNormals-v0_cpu_True",
# "/home/yixuan/EECS598_006_final_project/test_output/time_test/yixuan_21_05_24_02_26_01_horizon_5_solver_Normal_RRT_env_SShape-Boxes-256Pts-SurfaceNormals-v0_cpu_True"]
# f_list = ['test_output/connect_change/yixuan_21_07_06_01_00_58_horizon_5_solver_Normal_RRT_env_SShape-Boxes-1024Pts-SurfaceNormals-v0_cpu_True']
# f_list = ['test_output/it_comp/ensemble_boxes', 'test_output/it_comp/single_boxes']
# f_list = ['test_output/it_comp/ensemble_set_1', 'test_output/it_comp/single_set_1']
f_list = ['test_output/it_comp/RL_RRT_hardest', 'test_output/it_comp/Normal_RRT_hardest']

hard_seeds = [ 0, 2, 6, 8, 9, 10, 12, 16, 17, 24, 25, 26, 27, 35 ,36, 37, 39, 41, 43, 44, 45, 47, 50, 51, 53,
               62, 63, 65, 66, 69, 72, 73, 74, 76, 77, 78, 79, 81, 83, 85, 86, 90, 91, 92, 93, 96, 99]

# key = "iterations:"
key = "time"

data_np_list = []

for filename in f_list:
    print(filename)
    data_list = []
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == key:
            data_list.append(float(x[-1]))
    data_np = np.array(data_list)
    data_np = data_np[hard_seeds]

    data_np_list.append(data_np)

    print("mean:", data_np.mean())
    print("std:", data_np.std())

    # plt.figure()
    # plt.hist(data_np, bins=20, range=(0, 80000))
    # plt.xlabel("time(s)")
    # plt.ylabel("number of examples")
    # plt.savefig(filename.replace('/', '_')+'.png')

plt.figure()
plt.scatter(data_np_list[0], data_np_list[1])
plt.xlim(0, 80000)
plt.ylim(0, 80000)
plt.xlabel("RL RRT time")
plt.ylabel("Normal RRT time")
plt.savefig('compare.png')
