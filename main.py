import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from Normalize import norm
from network import Neural_Network
from predict import predict_nn
from one_hot import one_hot
data = scio.loadmat('data/ex3data1.mat')
X = data['X']
Y = data['y']
print("X.shape", X.shape)
print("Y.shape", Y.shape)

classes_num = 10
X_train = norm(X)
Y_train = one_hot(Y, classes_num)
learning_rate = 1
epochs = 1000
model = Neural_Network()
model.train(X_train, Y_train, learning_rate, epochs)
theta1 = model.theta1
theta2 = model.theta2
P = predict_nn(X_train, theta1, theta2)
Y_digit = Y.flatten() % 10
print(f"Y_digit shape {Y_digit.shape}")
print(f"Accuracy = {np.mean(P == Y_digit)*100}")