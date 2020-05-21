import  torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


def dataSet(n):
    x1_ = np.array([i / n for i in range(n)])
    x2_ = np.random.rand(n, )
    y_ = [1 if (x2_[i] + 0.02) > x1_[i] else -1 for i in range(n)]
    return x1_, x2_, y_

