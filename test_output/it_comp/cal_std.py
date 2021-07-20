import numpy as np
import matplotlib.pyplot as plt

std_list=[]
with open('debug.txt', 'r') as f:
    for line in f.readlines():
        s = float(line)
        print(s)
        std_list.append(s)
print("mean:", np.mean(std_list))
print("std:", np.std(std_list))

plt.hist(std_list, bins=10)
plt.xlabel("action std")
plt.ylabel("number of examples")
plt.savefig('ensemble_std.png')
