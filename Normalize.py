import numpy as np
def norm(x):
    x_norm = (x - np.mean(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0) + 1e-8)
    x_norm = np.column_stack((np.ones(x_norm.shape[0]), x_norm))
    return x_norm