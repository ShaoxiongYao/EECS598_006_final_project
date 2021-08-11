import numpy as np
import matplotlib.pyplot as plt

f_list = ['/home/yixuan/EECS598_006_final_project/test_output/0808test/yixuan_21_08_11_11_33_14_horizon_80_solver_Normal_RRT_env_SShape-Boxes-1024Pts-SurfaceNormals-v0_obs_boxes',
        '/home/yixuan/EECS598_006_final_project/test_output/0808test/yixuan_21_08_11_12_13_58_horizon_80_solver_single_RL_RRT_env_SShape-Boxes-1024Pts-SurfaceNormals-v0_obs_boxes',
        '/home/yixuan/EECS598_006_final_project/test_output/0808test/yixuan_21_08_11_12_30_07_horizon_80_solver_RL_RRT_env_SShape-Boxes-1024Pts-SurfaceNormals-v0_obs_boxes']

key_list = ["time", "time", "parallel"]
# key = "time"

for i, filename in enumerate(f_list):
    length = []
    f = open(filename, "r")
    for line in f:
        x=line.split()
        if x[0] == key_list[i]:
            length.append(float(x[-1]))
    # print(length)
    length_np = np.array(length)
    print(length_np.mean())
    plt.figure()
    plt.hist(length_np)
    plt.savefig(filename[:-4]+".png")