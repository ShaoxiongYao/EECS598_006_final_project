import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt("iteration_comparison.txt")

our = a[:,1]
rrt = a[:,2]
our_valid = our[rrt>0]
rrt_valid = rrt[rrt>0]

plt.figure()
rrt_hist = plt.hist(rrt_valid)
plt.xlabel("# iterations")
plt.ylabel("# cases")
plt.title("Histogram for Bi-RRT")
plt.savefig("rrt_dist.png")
our_hist = plt.hist(our_valid)
plt.title("Histogram for Bi-RRT and our method")
plt.legend(["Bi-RRT", "Our method"])
plt.savefig("combined.png")

plt.figure()
our_hist = plt.hist(our_valid)
plt.xlabel("# iterations")
plt.ylabel("# cases")
plt.title("Histogram for our method")
plt.savefig("our_dist.png")

print("mean of rrt iterations: ", np.mean(rrt_valid))

print("mean of our iterations: ", np.mean(our_valid))
