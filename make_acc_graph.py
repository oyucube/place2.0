import matplotlib.pyplot as plt
import numpy as np

acc_list = ["scale20_test", "scale32_s4_test"
            , "scale20_train", "scale32_s4_train"]

plt.figure()
plt.xlim([0, 20])
for item in acc_list:
    acc = np.load("buf/" + item + ".npy")
    plt.plot(acc, label=item)
#plt.legend(loc="lower left")
plt.legend()
plt.savefig("buf/graph.png")
