import numpy as np
import matplotlib.pyplot as plt

# a = np.loadtxt("iteration_comparison.txt")

# our = a[:,1]
# rrt = a[:,2]
# our_valid = our[rrt>0]
# rrt_valid = rrt[rrt>0]

our_txt = "RL_RRT_length_both_success.txt"
rrt_txt = "Normal_RRT_length.txt"
xlabel = "length"
ylabel = "# cases"
title1 = "Histgram for two methods"
pngname1 = "exp1_length_comb.png"

our_valid = np.loadtxt(our_txt)
rrt_valid = np.loadtxt(rrt_txt)
comb = np.zeros(shape=(our_valid.shape[0], 2))
comb[:, 0] = our_valid
comb[:, 1] = rrt_valid

# bins = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

plt.figure()
rrt_hist = plt.hist(comb, label=("our method", "Bi-RRT"))
# plt.xlabel("# iterations")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title1)
plt.legend()
plt.savefig(pngname1)

# plt.figure()
# rrt_hist = plt.hist(rrt_valid, bins=bins)
# plt.xlabel("# iterations")
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.title("Histogram for Bi-RRT")
# plt.savefig("rrt_dist.png")

# plt.figure()
# our_hist = plt.hist(our_valid, bins=bins)
# plt.xlabel("# iterations")
# plt.xlabel("time, s")
# plt.ylabel("# cases")
# plt.title("Histogram for our method")
# plt.savefig("our_dist.png")

print("mean of rrt: ", np.mean(rrt_valid))

print("mean of our: ", np.mean(our_valid))
