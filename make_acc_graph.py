import matplotlib.pyplot as plt
import numpy as np

acc_list = ["scale20_c15_test", "test2a32_test", "test2a64_test"]

plt.figure()
plt.xlim([0, 50])
for item in acc_list:
    acc = np.load("buf/" + item + ".npy")
    plt.plot(acc, label=item)
#plt.legend(loc="lower left")
plt.legend()
plt.savefig("buf/graph.png")
