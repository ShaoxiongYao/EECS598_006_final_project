import numpy as np

# filename = 'test_output/connect_change/start_q0.txt'
# average time: 0.6472502887839138
filename = 'test_output/connect_change/use_straight_line.txt'
# average time: 0.6832773850695921

time_list = []
with open(filename, 'r') as f:
    for line in f.readlines():
        time = float(line[14:])
        time_list.append(time)

print("average time:", np.mean(time_list))