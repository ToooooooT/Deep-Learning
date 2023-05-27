import numpy as np

class optimizer():
    def __init__(self, lr, *weights) -> None:
        self.lr = lr
        self.weights = list(weights)

    def update(self, *gradients):
        for i, grad in enumerate(gradients):
            self.weights[i] -= (grad * self.lr)
            

class SGD(optimizer):
    def __init__(self, lr, *weights) -> None:
        super().__init__(lr, *weights)

    def update(self, *gradients):
        return super().update(*gradients)


class SGDM(optimizer):
    def __init__(self, lr, Lambda=0.9, *weights) -> None:
        super().__init__(lr, *weights)
        self.Lambda = Lambda
        self.movements = list()
        for w in weights:
            self.movements.append(np.zeros(w.shape))

    def update(self, *gradients):
        '''
        v1 = lambda * v0 - lr * grad
        theta1 = theta0 + v1 
        '''
        for i, grad in enumerate(gradients):
            self.movements[i] = self.movements[i] * self.Lambda - self.lr * grad
            self.weights[i] += self.movements[i]


class NAG(optimizer):
    def __init__(self, lr, *weights) -> None:
        super().__init__(lr, *weights)


class Adagrad(optimizer):
    def __init__(self, lr, *weights) -> None:
        super().__init__(lr, *weights)
        self.total_g = list()
        for w in weights:
            self.total_g.append(np.zeros(w.shape))

    def update(self, *gradients):
        '''
        theta_t = theta_t-1 - lr / (sqrt(sum((g_i)^2) + epsilon) * g_t-1
        epsilon = 1e-8 to avoid divide by 0
        '''
        for i, grad in enumerate(gradients):
            self.total_g[i] += np.square(grad)
            self.weights[i] -= (self.lr / (np.sqrt(self.total_g[i]) + 1e-8) * grad)


class Adadelta(optimizer):
    def __init__(self, lr, rho=0.9, *weights) -> None:
        super().__init__(lr, *weights)
        self.rho = rho
        self.total_g = list()
        self.total_x = list()
        self.last_x = list()
        for w in weights:
            self.total_g.append(np.zeros(w.shape))
            self.total_x.append(np.zeros(w.shape))
            self.last_x.append(np.zeros(w.shape))

    def update(self, *gradients):
        '''
        E[g^2]_t+1 = rho * E[g^2]_t + (1 - rho) * (grad_t+1 ^ 2)
        RMS(g)_t+1 = sqrt(E[g^2]_t+1 + 1e-8)
        E[x]_t = rho * E[x^2]_t-1 + (1 - rho) * (x_t-1 ^ 2)
        RMS(x)_t = sqrt(E[x^2]_t + 1e-8)
        x = -g_t+1 * (RMS(x)_t / RMS(g)_t+1)
        '''
        epsilon = 1e-8
        for i, grad in enumerate(gradients):
            self.total_g[i] = self.rho * self.total_g[i] + (1 - self.rho) * np.square(grad)
            dominator = np.sqrt(self.total_g[i] + epsilon)
            self.total_x[i] = self.rho * self.total_x[i] + (1 - self.rho) * np.square(self.last_x[i])
            numerator= np.sqrt(self.total_x[i] + epsilon)
            x = -grad * numerator / dominator
            self.last_x[i] = x
            self.weights[i] += x


class Rmsprop(optimizer):
    def __init__(self, lr, alpha=0.9, *weights) -> None:
        super().__init__(lr, *weights)
        self.alpha = alpha
        self.movements = list()
        self.is_first_update = True
        for w in weights:
            self.movements.append(np.zeros(w.shape))

    def update(self, *gradients):
        '''
        theta_t = theta_t-1 - lr / sqrt(v_t) * g_t-1
        v_1 = (g_0)^2
        v_t = alpha * v_t-1 + (1 - alpha) * (g_t-1) ^ 2
        '''
        for i, grad in enumerate(gradients):
            if self.is_first_update:
                self.movements[i] = np.square(grad)
            else:
                self.movements[i] = self.alpha * self.movements[i] + (1 - self.alpha) * np.square(grad)
            self.weights[i] -= (self.lr / np.sqrt(self.movements[i]) * grad)
        
        self.is_first_update = False