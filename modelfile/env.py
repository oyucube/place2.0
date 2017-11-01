import numpy as np

import socket
xp = np
test = "cpu"
if socket.gethostname() == "naruto":
    import cupy as cp
    xp = cp
    test = "gpu"
