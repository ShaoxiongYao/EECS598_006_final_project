import json
import numpy as np
import matplotlib.pyplot as plt

RL_RRT_stats = {}
with open('RL_RRT_stats.json', 'r') as f:
    RL_RRT_stats = json.load(f)
Normal_RRT_stats = {}
with open('Normal_RRT_stats.json', 'r') as f:
    Normal_RRT_stats = json.load(f)

num_tests = len(RL_RRT_stats['time'])
non_trivial_idx = []
for idx in range(num_tests):
    if RL_RRT_stats['iterations'][idx] >= 1:
        non_trivial_idx.append(idx)

comb = np.zeros(shape=(num_tests, 2))
comb[:, 0] = RL_RRT_stats['time']
comb[:, 1] = Normal_RRT_stats['time']

plt.hist(comb[non_trivial_idx, :], range=(0, 180), label=['RL_RRT', 'Normal_RRT'])
plt.xlabel("time(s)")
plt.ylabel("number of examples")
plt.title("Time distribution")
plt.savefig('time_distribution.png')
