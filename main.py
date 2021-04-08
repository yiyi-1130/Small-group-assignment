import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
def LinerRegression(X,Y):
    X_b=np.c_[np.ones((len(X), 1)), X]
    theta= np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
    Y_predict = X_b.dot(theta)
    score = 1 - mean_squared_error(Y, Y_predict) / np.var(Y)
    return X_b, theta, score

if __name__=='__main__':
    X = np.random.rand(100, 1)
    Y = 4+3*X+np.random.rand(100, 1)
    X_b, theta, score = LinerRegression(X, Y)
    Y_predict=X_b.dot(theta)
    print("回归直线对观测值的拟合程度:")
    print(score)
    plt.plot(X, Y, 'o')
    plt.plot(X, Y_predict, 'r')
    plt.show()