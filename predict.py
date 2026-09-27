import numpy as np
from activation import sigmoid
def predict_nn(X, theta1, theta2):
    print(f"Theta 1 shape = {theta1.shape}")
    print(f"Theta 2 shape = {theta2.shape}")
    Z1 = np.dot(theta1, X.T)
    A1 = sigmoid(Z1)                                   
    A1 = np.row_stack((np.ones(A1.shape[1]), A1))        
    Z2 = np.dot(theta2, A1)
    A2 = sigmoid(Z2.T)
    Pred = np.argmax(A2, axis=1)
    return Pred