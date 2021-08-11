import numpy as np
import matplotlib.pyplot as plt

filenames=["/home/yixuan/EECS598_006_final_project/test_output/diversity_test/boxes.txt",
            "/home/yixuan/EECS598_006_final_project/test_output/diversity_test/ycb.txt",
            "/home/yixuan/EECS598_006_final_project/test_output/diversity_test/handcraft_hardest.txt"]
for fn in filenames:
    diversity = np.loadtxt(fn)
    print("diversity mean:", diversity.mean())
    print("diversity mean+stddev:", diversity.mean()+diversity.std())
    plt.figure()
    plt.hist(diversity)
    plt.savefig(fn[:-4]+".png")