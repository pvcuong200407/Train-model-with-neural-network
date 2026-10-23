import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from display_data import display_data

def split_stratifided (X, y, ratio_learn = 0.8):
    np.random.seed(1)
    y = y.flatten() % 10
    test_idx = []
    train_idx = []
    for c in range(10):
        idx = np.where(y==c)[0]
        np.random.shuffle(idx)
        n_test = int(len(idx)*ratio_learn)
        train_idx += list(idx[:n_test])
        test_idx += list(idx[n_test:])
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def onehot(y, classes_num):
    y = y.flatten()
    y = y % 10
    y_oh = np.zeros((y.size, classes_num))
    y_oh[np.arange(y.size), y] = 1
    return y_oh

def fit_norm(x):
    return np.mean(x, axis=0), np.max(x, axis=0) - np.min(x, axis=0) + 1e-8

def apply_norm(x, mean, rng_):
    x_norm = (x - mean) / rng_
    return np.column_stack((np.ones(x_norm.shape[0]), x_norm))

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def ReLu(z):
    return np.where(z > 0, z, 0.01 * z)

def grad_relu(z):
    return np.where(z > 0, 1.0, 0.01)

def SoftMax(z):
    z = z - np.max(z, axis = 1, keepdims= True)
    e = np.exp(z)
    return e / np.sum(e, axis =1, keepdims = True)
# neural_network: snake case (local_variable, global_variable)
# NeuralNetwork:  camel case (ClassName, StructName)
# Example:
#   net = NeuralNetwork()
#   net = NeuralNetwork(100)
#   net = NeuralNetwork(10,hidden_state=32)

class NeuralNetwork:

    def __init__(self, image_size, classes_num, hidden_state=25):
        np.random.seed(1)
        self.theta1 = np.random.randn(hidden_state, image_size**2+1) * 0.01
        self.theta2 = np.random.randn(classes_num, hidden_state+1  ) * 0.01
        self.loss = []
        self.lr   = []
        self.classes_num  = classes_num
        self.hidden_state = hidden_state

    def forward(self, X):
        self.Z1 = np.dot(self.theta1, X.T)
        A1 = ReLu(self.Z1)       # best practice : relu
        self.A1 = np.vstack((np.ones(A1.shape[1]), A1))
        Z2 = np.dot(self.theta2, self.A1)
        self.A2 = SoftMax(Z2.T) # best pracetice: softmax for multi-label, sigmoid for binary classification
        return self.A2

    def backward(self, X, Y):
        self.m = X.shape[0]
        dZ2 = self.A2 - Y                                   
        self.dtheta2 = (self.A1 @ dZ2).T                     
        dA1 = (dZ2 @ self.theta2)[:, 1:].T                   
        dZ1 = dA1 * grad_relu(self.Z1)                       
        self.dtheta1 = dZ1 @ X    
        

    def update(self, learning_rate):
        self.theta2 -= (1/self.m)*learning_rate*(self.dtheta2)
        self.theta1 -= (1/self.m)*learning_rate*(self.dtheta1)

    def train(self, X, Y, learning_rate, epochs):
        self.loss = []
        for _ in range(epochs):
            y_pred = self.forward(X)
            self.loss.append(-np.mean(np.sum(Y * np.log(y_pred + 1e-12), axis=1)))
            self.backward(X, Y)
            self.update(learning_rate)

    def predict(self, X):
        print(f"Theta 1 shape = {self.theta1.shape}")
        print(f"Theta 2 shape = {self.theta2.shape}")
        Z1 = np.dot(self.theta1, X.T)
        A1 = ReLu(Z1)                                   
        A1 = np.vstack((np.ones(A1.shape[1]), A1))        
        Z2 = np.dot(self.theta2, A1)
        A2 = SoftMax(Z2.T)
        pred = np.argmax(A2, axis=1)
        return pred

    def plot(self, X_train, Y_train, lrs, classes_num, epochs):
    #     # TODO : Plot loss, learning rate
        plt.figure()
        for lr in lrs:
            model = NeuralNetwork(20, classes_num)   
            model.train(X_train, Y_train, lr, epochs)
            plt.plot(model.loss, label=f"lr = {lr}")
        
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss theo learning rate")
        plt.legend()
        plt.show()
        

def main():

    # pythonic function definition
    
    # used in binary classification / multi-label classification
    # sigmoid = lambda g : 1/(1+np.exp(-z))

    # multi-class classification
    # last layer uses softmax

    # TODO : visulize your input data

    data = scio.loadmat('data/ex3data1.mat')
    X = data['X']
    Y = data['y']
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)

    ratio_learn = 0.8
     # TODO : split dataset to X_test, Y_test
    X_train_raw, y_train, X_test_raw, y_test = split_stratifided(X, Y, ratio_learn)
    classes_num = 10
    
    # Pre-processing
    mean, scale = fit_norm(X_train_raw)
    X_train = apply_norm(X_train_raw, mean, scale)
    X_test = apply_norm(X_test_raw, mean, scale)
    y_train_oh = onehot(y_train, classes_num)
   

    learning_rate = 1
    epochs = 1000
    model = NeuralNetwork(20, classes_num)
    model.train(X_train, y_train_oh, learning_rate, epochs)
    P = model.predict(X_test)
    print(f"P.shape = {P.shape}")
    print(f"Accuracy = {np.mean(P == y_test)*100}")

    # TODO : visulize your training curve
    # model.plot()
    lrs = np.linspace(0.1,1,10)
    model.plot(X_train, y_train, lrs, classes_num, 300)
    m = X_test.shape[0]
    rp = np.random.permutation(m)
    for i in range (m):
        example = X_test_raw[rp[i]].reshape(1, -1)             
        example_norm = apply_norm(example, mean, scale)    

        pred = model.predict(example_norm)[0]
        print(f"Dự đoán: {pred} | thật: {y_test[rp[i]]}")
        display_data(example)                             
        s = input('Paused - press ENTER to continue, q + ENTER to exit: ')
        if s == 'q':
            break
if __name__ == "__main__":
    main()


#  Next steps:
#     - Convolution Neural Network (CNN)
#     - Recurrent Neural Network   (RNN)
#     - Attention (Transformers)