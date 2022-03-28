import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Pollution = pd.read_csv("D:\yadav\Coding Blocks\Machine Learning\Challenge - Air Pollution Regression\Train.csv")
Y = Pollution.target

X = Pollution.drop("target",axis=1)

u = X.mean()
std = X.std()

X = (X-u)/std
ones = np.ones((X.shape[0],1))
X = np.hstack((ones,X))


def hypothesis(X,theta):
    return np.dot(X,theta)

def error(X,Y,theta):
    m = X.shape[0]
    e = 0.0
    y_ = hypothesis(X,theta)
    e = np.sum((y_-Y)**2)
    return e/m

def Gradient(X,Y,theta):
    m = X.shape[0]
    y_ = hypothesis(X,theta)
    grad = np.dot(X.T,(y_-Y))
    return grad/m


def Gradient_descent(X,Y,max_steps=100,learning_rate = 0.1):
    n = X.shape[1]
    error_list = []
    theta = np.zeros((n,))
    for i in range(max_steps):
        e = error(X,Y,theta)
        error_list.append(e)
        grad = Gradient(X,Y,theta)
        theta = theta-learning_rate*grad
    return theta,error_list


theta,error_list = Gradient_descent(X,Y)

def r2_score(Y,Y_):
    num = np.sum(((Y-Y_)**2))
    den = np.sum((Y-Y.mean())**2)
    r2 = (1-num/den)
    return r2*100

y_ = hypothesis(X,theta)
print(r2_score(Y,y_))