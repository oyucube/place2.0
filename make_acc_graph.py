import matplotlib.pyplot as plt
import numpy as np

mean_list = ["try1", "try2", "try3", "try4", "try5"]
acc_list = ["scale_test", "no_locate_info_test", "change_RL_0.2_test"]

arr = np.zeros((5, 30))
i = 0
for item in mean_list:
    acc = np.load("graph/" + item + ".npy")
    arr[i] = acc
    i += 1
print(arr)

mean = arr.mean(axis=0)
std = arr.std(axis=0)
print("mean")
print(mean)

print("std")
print(std)


plt.figure()
plt.xlim([0, 30])
plt.plot(arr[0], label="pre")
plt.errorbar(range(30), mean, yerr=std, label="average")
plt.legend()
plt.savefig("graph/test.png")

plt.figure()
plt.rcParams["font.size"] = 18
plt.xlim([0, 30])
plt.ylim([0, 1])
# plt.errorbar(range(30), mean, yerr=std, label="average_test")
for item in acc_list:
    acc = np.load("graph/" + item + ".npy")
    plt.plot(acc, label=item)
#plt.legend(loc="lower left")
plt.legend()
plt.savefig("graph/graph.png")
