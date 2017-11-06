import matplotlib.pyplot as plt
import numpy as np

acc_list = ["scale20_test", "p1_test", "p2_test", "p2a_test", "p2b_test", "bnlstm_test", "p1m20_test", "p2m20_test"]

plt.figure()
plt.xlim([0, 30])
for item in acc_list:
    acc = np.load("buf/" + item + ".npy")
    plt.plot(acc, label=item)
#plt.legend(loc="lower left")
plt.legend()
plt.savefig("buf/graph.png")
