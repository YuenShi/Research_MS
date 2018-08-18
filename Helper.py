# Author： 施源 Kris
# Create Time： 2018.8.15

import matplotlib.pyplot as plt
import numpy as np

# 画图查看结果
def plot_fortune(data):
    plt.plot(np.arange(len(data)), data)
    plt.ylabel('Fortunes')
    plt.xlabel('Training steps')
    plt.show()


