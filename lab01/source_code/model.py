import numpy as np
from optimizer import *

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def ReLU(x):
    tmp = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp[i, j] = x[i, j] if x[i, j] > 0 else 0 
    return tmp

def derivative_ReLU(x):
    tmp = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp[i, j] = 1 if x[i, j] > 0 else 0 
    return tmp

def LeakyReLU(x, alpha=0.01):
    tmp = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp[i, j] = x[i, j] if x[i, j] > 0 else alpha * x[i, j] 
    return tmp

def derivative_LeakyReLU(x, alpha=0.01):
    tmp = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp[i, j] = 1 if x[i, j] > 0 else alpha
    return tmp

class Model():
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, lr=0.001, activation=0) -> None:
        '''
        activation:
            0 : Sigmoid
            1 : ReLU
            2 : LeakyReLU
        '''
        self.activation = activation
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        self.W1 = np.random.normal(0, 0.1, (self.hidden1_size, input_size))
        self.W2 = np.random.normal(0, 0.1, (self.hidden2_size, self.hidden1_size))
        self.W3 = np.random.normal(0, 0.1, (output_size, self.hidden2_size))

        self.b1 = np.random.normal(0, 0.1, (self.hidden1_size, 1))
        self.b2 = np.random.normal(0, 0.1, (self.hidden2_size, 1))
        self.b3 = np.random.normal(0, 0.1, (output_size, 1))

        self.lr = lr
        self.optimizer = SGD(lr, self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)
        # self.optimizer = SGDM(lr, 0.9, self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)
        # self.optimizer = Adagrad(lr, self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)
        # self.optimizer = Adadelta(lr, 0.9, self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)
        # self.optimizer = Rmsprop(lr, 0.9, self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)

    def cross_entropy_loss(self, y_pred, y_hat):
        self.y_pred = y_pred
        self.y_hat = y_hat
        loss = np.zeros((1, y_pred.shape[1]))
        for i in range(y_pred.shape[1]):
            if y_pred[0, i] == 0 and y_hat[0, i] == 1:
                loss[0, i] = 100 
            elif y_pred[0, i] == 1 and y_hat[0, i] == 0: 
                loss[0, i] = 100
            elif y_pred[0, i] != 0 and y_pred[0, i] != 1:
                loss[0, i] = -(y_hat[0, i] * np.log(y_pred[0, i]) + (1 - y_hat[0, i]) * np.log(1 - y_pred[0, i]))
        return loss.mean()

    def cross_entropy_grad(self):
        loss_y_der = np.zeros((1, self.y_pred.shape[1]))
        for i in range(self.y_pred.shape[1]):
            if self.y_pred[0, i] == 0 and self.y_hat[0, i] == 1:
                loss_y_der[0, i] = -100 
            elif self.y_pred[0, i] == 1 and self.y_hat[0, i] == 0: 
                loss_y_der[0, i] = -100
            elif self.y_pred[0, i] != 0 and self.y_pred[0, i] != 1:
                loss_y_der[0, i] = -self.y_hat[0, i] / self.y_pred[0, i] + (1 - self.y_hat[0, i]) / (1 - self.y_pred[0, i])
        return loss_y_der

    def forward(self, input):
        self.input = input                                          # (2, batch_size)
        if self.activation == 0:
            self.A1 = sigmoid(self.W1 @ self.input + self.b1)       # (l1, batch_size)
            self.A2 = sigmoid(self.W2 @ self.A1 + self.b2)          # (l2, batch_size)
        elif self.activation == 1:
            self.A1 = ReLU(self.W1 @ self.input + self.b1)          # (l1, batch_size)
            self.A2 = ReLU(self.W2 @ self.A1 + self.b2)             # (l2, batch_size)
        else:
            self.A1 = LeakyReLU(self.W1 @ self.input + self.b1)     # (l1, batch_size)
            self.A2 = LeakyReLU(self.W2 @ self.A1 + self.b2)        # (l2, batch_size)
        self.y = sigmoid(self.W3 @ self.A2 + self.b3)               # (1, batch_size)
        return self.y
    
    def backward(self):
        if self.activation == 0:
            activation1_der = derivative_sigmoid(self.A1)   # (l1, batch_size)
            activation2_der = derivative_sigmoid(self.A2)   # (l2, batch_size)
        elif self.activation == 1:
            activation1_der = derivative_ReLU(self.A1)      # (l1, batch_size)
            activation2_der = derivative_ReLU(self.A2)      # (l2, batch_size)
        else:
            activation1_der = derivative_LeakyReLU(self.A1) # (l1, batch_size)
            activation2_der = derivative_LeakyReLU(self.A2) # (l2, batch_size)
        activation3_der = derivative_sigmoid(self.y)        # (1, batch_size)

        loss_y_der = self.cross_entropy_grad()

        Z3_der = loss_y_der * activation3_der               # (1, batch_size)
        Z2_der = activation2_der * (self.W3.T @ Z3_der)     # (l2, batch_size)
        Z1_der = activation1_der * (self.W2.T @ Z2_der)     # (l1, batch_size)

        self.W3_grad = Z3_der @ (self.A2.T)     # (1, l2)
        self.W2_grad = Z2_der @ (self.A1.T)     # (l2, l1)
        self.W1_grad = Z1_der @ (self.input.T)  # (l1, 2)

        self.b3_grad = Z3_der.sum(axis=1) 
        self.b2_grad = Z2_der.sum(axis=1) 
        self.b1_grad = Z1_der.sum(axis=1) 

    def update(self):
        self.optimizer.update(self.W1_grad, self.W2_grad, self.W3_grad, self.b1, self.b2, self.b3)


class ConvModel():
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, lr=0.001, activation=0) -> None:
        '''
        activation:
            0 : Sigmoid
            1 : ReLU
            2 : LeakyReLU
        '''
        self.activation = activation
        self.kernel_size = (1, 2)
        self.channel = 10
        self.hidden1_size = (input_size + 3 - self.kernel_size[1]) * self.channel
        self.hidden2_size = hidden2_size

        self.W1 = np.random.normal(0, 0.1, (self.kernel_size[0], self.kernel_size[1], self.channel))
        self.W2 = np.random.normal(0, 0.1, (self.hidden2_size, self.hidden1_size))
        self.W3 = np.random.normal(0, 0.1, (output_size, self.hidden2_size))

        self.b1 = np.random.normal(0, 0.1, (self.channel, 1))
        self.b2 = np.random.normal(0, 0.1, (self.hidden2_size, 1))
        self.b3 = np.random.normal(0, 0.1, (output_size, 1))

        self.lr = lr
        self.optimizer = SGD(lr, self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)
        # self.optimizer = SGDM(lr, 0.9, self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)
        # self.optimizer = Adagrad(lr, self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)
        # self.optimizer = Adadelta(lr, 0.9, self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)
        # self.optimizer = Rmsprop(lr, 0.9, self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)

    def cross_entropy_loss(self, y_pred, y_hat):
        self.y_pred = y_pred
        self.y_hat = y_hat
        loss = np.zeros((1, y_pred.shape[1]))
        for i in range(y_pred.shape[1]):
            if y_pred[0, i] == 0 and y_hat[0, i] == 1:
                loss[0, i] = 100 
            elif y_pred[0, i] == 1 and y_hat[0, i] == 0: 
                loss[0, i] = 100
            elif y_pred[0, i] != 0 and y_pred[0, i] != 1:
                loss[0, i] = -(y_hat[0, i] * np.log(y_pred[0, i]) + (1 - y_hat[0, i]) * np.log(1 - y_pred[0, i]))
        return loss.mean()

    def cross_entropy_grad(self):
        loss_y_der = np.zeros((1, self.y_pred.shape[1]))
        for i in range(self.y_pred.shape[1]):
            if self.y_pred[0, i] == 0 and self.y_hat[0, i] == 1:
                loss_y_der[0, i] = -100 
            elif self.y_pred[0, i] == 1 and self.y_hat[0, i] == 0: 
                loss_y_der[0, i] = -100
            elif self.y_pred[0, i] != 0 and self.y_pred[0, i] != 1:
                loss_y_der[0, i] = -self.y_hat[0, i] / self.y_pred[0, i] + (1 - self.y_hat[0, i]) / (1 - self.y_pred[0, i])
        return loss_y_der

    def forward(self, input):
        zero_rows = np.zeros((1, input.shape[1]))
        self.input = np.vstack((zero_rows, input, zero_rows))       # (4, batch_size), padding with zeros
        self.A1 = np.zeros((1, self.input.shape[1]))
        for c in range(self.W1.shape[2]):
            for i in range(self.input.shape[0] - self.W1.shape[1] + 1):
                self.A1 = np.vstack((self.A1, self.W1[:, :, c] @ self.input[i:i+self.W1.shape[1], :] + self.b1[c]))
        self.A1 = np.delete(self.A1, 0, axis=0)
        if self.activation == 0:
            self.A1 = sigmoid(self.A1)                              # (l1, batch_size)
            self.A2 = sigmoid(self.W2 @ self.A1 + self.b2)          # (l2, batch_size)
        elif self.activation == 1:
            self.A1 = ReLU(self.A1)                                 # (l1, batch_size)
            self.A2 = ReLU(self.W2 @ self.A1 + self.b2)             # (l2, batch_size)
        else:
            self.A1 = LeakyReLU(self.A1)                            # (l1, batch_size)
            self.A2 = LeakyReLU(self.W2 @ self.A1 + self.b2)        # (l2, batch_size)
        self.y = sigmoid(self.W3 @ self.A2 + self.b3)               # (1, batch_size)
        return self.y
    
    def backward(self):
        if self.activation == 0:
            activation1_der = derivative_sigmoid(self.A1)   # (l1, batch_size)
            activation2_der = derivative_sigmoid(self.A2)   # (l2, batch_size)
        elif self.activation == 1:
            activation1_der = derivative_ReLU(self.A1)      # (l1, batch_size)
            activation2_der = derivative_ReLU(self.A2)      # (l2, batch_size)
        else:
            activation1_der = derivative_LeakyReLU(self.A1) # (l1, batch_size)
            activation2_der = derivative_LeakyReLU(self.A2) # (l2, batch_size)
        activation3_der = derivative_sigmoid(self.y)        # (1, batch_size)

        loss_y_der = self.cross_entropy_grad()

        Z3_der = loss_y_der * activation3_der               # (1, batch_size)
        Z2_der = activation2_der * (self.W3.T @ Z3_der)     # (l2, batch_size)
        Z1_der = activation1_der * (self.W2.T @ Z2_der)     # (l1, batch_size)

        self.W3_grad = Z3_der @ (self.A2.T)     # (1, l2)
        self.W2_grad = Z2_der @ (self.A1.T)     # (l2, l1)

        self.W1_grad = np.zeros((self.W1.shape[0], self.W1.shape[1], self.W1.shape[2]))
        n = self.input.shape[0] - self.W1.shape[1] + 1
        for c in range(self.W1.shape[2]):
            for i in range(self.W1.shape[1]):
                self.W1_grad[0, i, c] = (Z1_der[c * 3: c * 3 + n, :] * self.input[i:i + n, :]).sum().sum()

        self.b3_grad = Z3_der.sum(axis=1) 
        self.b2_grad = Z2_der.sum(axis=1) 

        self.b1_grad = np.zeros((self.b1.shape[0], self.b1.shape[1]))
        Z1_der = Z1_der.sum(axis=1)
        for i in range(self.b1.shape[0]):
            self.b1_grad[i] = Z1_der[i * n] + Z1_der[i * n + 1] + Z1_der[i * n + 2]

    def update(self):
        self.optimizer.update(self.W1_grad, self.W2_grad, self.W3_grad, self.b1, self.b2, self.b3)