filename = "/home/yixuan/EECS598_006_final_project/test_output/yixuan_21_04_29_04_21_18_horizon_5_solver_Normal_RRT_cpu_True"
f = open(filename, "r")
for line in f:
    x=line.split()
    if x[0] == "time":
        print(x[-1])