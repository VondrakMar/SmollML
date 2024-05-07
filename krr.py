import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class rbf_kernel():
    def __init__(self,gamma):
        self.gamma = gamma

    def calc_K(self,X1,X2):
        sq_dist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sq_dist) 

class kkr():
    def __init__(self,kernel,X_train=None,y_train=None):
        self.kernel = kernel
        self.X_train = X_train
        self.y_train = y_train
        self.weights = None

    def train(self,sigma,X_train=None, y_train=None):
        if X_train != None:
            self.X_train = X_train
            self.y_train = y_train
        K = self.kernel.calc_K(self.X_train, self.X_train)
        n_samples = K.shape[0]
        Sigma = sigma*np.eye(n_samples)
        self.weights = np.linalg.inv(K + Sigma).dot(self.y_train)

    def loss(self,params):
        w = params
        K = self.kernel.calc_K(self.X_train, self.X_train)
        y_pred = K.T.dot(w)
        mse = np.mean((self.y_train - y_pred) ** 2)
        return mse
        
    def train_minimize(self,sigma,X_train=None, y_train=None):
        if X_train != None:
            self.X_train = X_train
            self.y_train = y_train
        
        initial_weights = np.random.rand(len(self.y_train))
        weights_res = minimize(self.loss, initial_weights, method='L-BFGS-B')
        self.weights = weights_res.x
        
    def predict(self,X_test):
        K_test = self.kernel.calc_K(self.X_train, X_test)
        return K_test.T.dot(self.weights)
        
        
rng = np.random.default_rng(1)
X_train = np.linspace(0, 5, 100)[:, None]
y_train = np.sin(X_train).ravel() + 0.1 * rng.standard_normal(X_train.shape[0])
X_test = np.linspace(0, 5, 46)[:, None]
sigma = 0.1  
gamma = 0.5  


my_kernel = rbf_kernel(gamma)
my_model = kkr(my_kernel,X_train,y_train)
my_model.train(sigma)


y_pred = my_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='darkorange', label='Data')
plt.plot(X_test, y_pred, color='navy', lw=2, label='Kernel Ridge Regression')
plt.xlabel('Data')
plt.ylabel('Target')
plt.title('Kernel Ridge Regression from Scratch')

my_model.train_minimize(sigma)
y_pred = my_model.predict(X_test)
plt.plot(X_test, y_pred, color='green', lw=2, label='scipy')



plt.legend()
plt.show()
