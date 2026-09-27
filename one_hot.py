import numpy as np
def one_hot(y, classes_num):
    y = y.flatten()
    y = y % 10
    y_oh = np.zeros((y.size, classes_num))
    y_oh[np.arange(y.size), y] = 1
    return y_oh