import numpy as np
import matplotlib.pyplot as plt

class Linear_regression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w = np.random.randn(1,x.shape[0]) 
        self.b = np.random.randn(1,1)
    
    def forward_propagation(self):
        return (self.w).dot(self.x) + self.b

    def mse(self):
        n = self.y.shape[1]
        return (1/n) * np.sum(np.square(self.y - self.forward_propagation()))
    
    def optimisation(self, learning_rate):
        n= self.y.shape[1]
        error= self.y - self.forward_propagation()

        dw = -2/n * error.dot(self.x.T)
        db = -2/n * np.sum(error,axis=1, keepdims = True)

        self.w  = self.w - learning_rate * dw
        self.b = self.b - learning_rate * db
    
    def train(self, learning_rate = 0.01, iteration = 100):
        loss = []
        for i in range(iteration ):
            y_predict = self.forward_propagation()
            self.optimisation(learning_rate)
            if (i % 10==0):
                loss.append(self.mse())
        #plt.plot(loss)
        #print(self.w)
        return (self.w,self.b,loss)
    
    def predict(self, x):
        y_predict = (self.w).dot(x) + self.b
        return y_predict