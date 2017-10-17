import matplotlib.pyplot as plt
import numpy as np

acc_list = ["", ""]

plt.figure()
for item in acc_list:
    acc = np.load("buf/" + item)
    plt.plot(acc, label="item")
plt.legend()
plt.savefig("buf/graph.png")
