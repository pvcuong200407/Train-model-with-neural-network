import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g
