import matplotlib.pyplot as plt
import numpy as np

acc_list = ["vgg_train", "vgg_test", "dram_vgg_train", "dram_vgg_test", "normal_train", "normal_test"]

plt.figure()
plt.xlim([0, 20])
for item in acc_list:
    acc = np.load("buf/" + item + ".npy")
    plt.plot(acc, label=item)
plt.legend()
plt.savefig("buf/graph.png")
