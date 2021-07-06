import json
import numpy as np
import matplotlib.pyplot as plt

RL_RRT_stats = {}
with open('RL_RRT_stats.json', 'r') as f:
    RL_RRT_stats = json.load(f)
Normal_RRT_stats = {}
with open('Normal_RRT_stats.json', 'r') as f:
    Normal_RRT_stats = json.load(f)

fig1 = plt.figure()

num_tests = len(RL_RRT_stats['time'])
non_trivial_idx = []
for idx in range(num_tests):
    if RL_RRT_stats['iterations'][idx] >= 10:
        non_trivial_idx.append(idx)
print("non-trivial tests:", len(non_trivial_idx))

comb = np.zeros(shape=(num_tests, 2))
comb[:, 0] = RL_RRT_stats['time']
comb[:, 1] = Normal_RRT_stats['time']
non_trivial_comb = comb[non_trivial_idx, :]

plt.hist(non_trivial_comb, range=(0, 230), label=['RL_RRT', 'Normal_RRT'])
print("RL_RRT non-trivial mean and std time:", non_trivial_comb[:, 0].mean(), non_trivial_comb[:, 0].std())
print("Normal_RRT non-trivial mean and std time:", non_trivial_comb[:, 1].mean(), non_trivial_comb[:, 1].std())

plt.xlabel("time(s)")
plt.ylabel("number of examples")
plt.title("Time distribution")
plt.legend()
plt.savefig('time_distribution.png')

fig2 = plt.figure()

comb = np.zeros(shape=(num_tests, 2))
comb[:, 0] = RL_RRT_stats['iterations']
comb[:, 1] = Normal_RRT_stats['iterations']
non_trivial_comb = comb[non_trivial_idx, :]

plt.hist(non_trivial_comb, range=(0, 50000), label=['RL_RRT', 'Normal_RRT'])
print("RL_RRT non-trivial mean and std iterations:", non_trivial_comb[:, 0].mean(), non_trivial_comb[:, 0].std())
print("Normal_RRT non-trivial mean and std iterations:", non_trivial_comb[:, 1].mean(), non_trivial_comb[:, 1].std())

plt.xlabel("iterations")
plt.ylabel("number of examples")
plt.title("Iterations distribution")
plt.legend()
plt.savefig('iterations_distribution.png')
