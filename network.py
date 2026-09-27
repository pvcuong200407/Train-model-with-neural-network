import numpy as np
from activation import sigmoid
class Neural_Network:

    def __init__(self):
        np.random.seed(1)  
        self.theta1 = np.random.randn(25, 401) * 0.01
        self.theta2 = np.random.randn(10, 26) * 0.01
    def forward(self, X):
        Z1 = np.dot(self.theta1, X.T)
        A1 = sigmoid(Z1)
        self.A1 = np.row_stack((np.ones(A1.shape[1]), A1))
        Z2 = np.dot(self.theta2, self.A1)
        self.A2 = sigmoid(Z2.T)
        return self.A2
    def backward(self, X, Y):
        self.m = X.shape[0]
        dZ2 = (self.A2 - Y)
        self.dtheta2 = ((self.A1) @ dZ2).T
        dA1 = dZ2 @ self.theta2
        dA1 = dA1[:,1:].T
        A1_no_bias = self.A1[1:,:]
        dZ1 = dA1 * A1_no_bias * (1 - A1_no_bias)
        self.dtheta1 = dZ1 @ X

    def update(self, learning_rate):
        self.theta2 -= (1/self.m)*learning_rate*(self.dtheta2)
        self.theta1 -= (1/self.m)*learning_rate*(self.dtheta1)

    def train(self, X, Y, learning_rate, epochs):
        for i in range (epochs):
            A2 = self.forward(X)
            self.backward(X, Y)
            self.update(learning_rate)

            loss = -np.mean(Y*np.log(A2) + (1-Y)*np.log(1-A2))
            if (i % 100 ==0): 
                print("Epochs =",i, "\nLost =", loss)

