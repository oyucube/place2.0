import matplotlib.pyplot as plt
import numpy as np

acc_list = ["scale20_test", "dram20_test", "dram40_test", "dram80_test"]

plt.figure()
plt.xlim([0, 30])
for item in acc_list:
    acc = np.load("buf/" + item + ".npy")
    plt.plot(acc, label=item)
#plt.legend(loc="lower left")
plt.legend()
plt.savefig("buf/graph.png")
