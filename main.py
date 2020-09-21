# Reproduction of "Scalable Approximations for Generalized Linear Problems"
# by Murat A. Erdogdu, Mohsen Bayati, Lee H. Dicker

# Michael Wieck-Sosa
# Professor Fu
# Numerical Analysis
# December 8, 2019

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
import math
import time
from sklearn import linear_model
import datetime
import scipy
from mpmath import mp

class SLS_Estimator:
    def __init__(self,X,y,genFn,subsetDivisor,tol,max_iter):
        self.X = X
        self.y = y
        self.genFn = genFn
        self.tol = tol
        self.max_iter = max_iter
        self.n = self.X.shape[0]
        self.subsetDivisor = subsetDivisor
        self.ns = int(self.n / self.subsetDivisor)
        self.Xs = self.X[np.random.randint(self.X.shape[0], size = self.ns), :]
        self.B_OLS = ((self.ns / self.n)*
                     ((np.linalg.inv(self.Xs.T@self.Xs))@(self.X.T@self.y)))
        self.yhat = self.X@self.B_OLS

    def psi0(self, w):
        if genFn == "logistic":
            return math.log10(1 + np.exp(w))
        if genFn == "poisson":
            return np.exp(w)
        if genFn == "linear":
            return w**2 / 2

    def psi2(self, w):
        if self.genFn == "logistic":
            return np.exp(w) / (np.exp(w) + 1)**2
        if self.genFn == "poisson":
            return np.exp(w)
        if self.genFn == "linear":
            return 1

    def psi3(self, w):
        if self.genFn == "logistic":
            return np.exp(w) / (np.exp(w) + 1)**3
        if self.genFn == "poisson":
            return np.exp(w)
        if self.genFn == "linear":
            return 0

    def f(self, c):
        rsum = 0
        for i in range(self.n):
            rsum += self.psi2(c*self.yhat[i])
        return (c / self.n) * rsum - 1

    def derivative(self, c):
        rsum = 0
        for i in range(self.n):
            rsum += (self.psi2(c*self.yhat[i])+
                    c*self.yhat[i]*self.psi3(c*self.yhat[i]))
        return rsum / n

    def getSLS_Estimator(self):
        c = 1 #init for newton's method
        it = 0
        while abs(self.f(c)) > self.tol and it < self.max_iter:
            c = c - self.f(c) / self.derivative(c)
            it += 1
        return c, self.B_OLS # B_SLS = c * B_OLS


def cal_cost(theta,X,y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost

def gradient_descent(X, y, theta, learning_rate=1e-5, iterations=1000):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 200))
    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)
    return theta, cost_history, theta_history


for distr in ["logistic","linear","poisson"]:

    print("Generating results for",distr,"...")

    SLS_times = []
    GD_times = []

    SLS_absErr = []
    GD_absErr = []

    SLS_errUppBnd = []
    GD_errUppBnd = []

    SLS_errLwrBnd = []
    GD_errLwrBnd = []

    SLS_convergence = []

    n_list = [int(1e3),int(1e4),int(1e5),int(1e6)]
    for n in n_list:
        # Generate artificial data
        X = np.random.random((n, 200))
        beta = np.random.rand(200) # dim of feat space

        if distr == "logistic":
            mu, sigma = 0, 1 # mean, std dev
            err = np.random.normal(mu, sigma, n) # normal error
            pr = (np.exp(np.dot(X,beta)+err) / (np.add(1,np.exp(np.dot(X,beta)+err))))
            numTrials = 1
            y = np.random.binomial(numTrials, pr, n)

        elif distr == "linear":
            mu, sigma = 0, 1 # mean, std dev
            err = np.random.normal(mu, sigma, n) # normal error
            y = np.dot(X, beta) + err

        elif distr == "poisson":
            lambda_ = 1
            err = np.random.poisson(lambda_,n) # poisson error
            y = np.dot(X, beta) + err

        ### scaled least squared estimator ###
        start_time = datetime.datetime.now()
        class_SLS = SLS_Estimator(X=X,y=y,genFn=distr,subsetDivisor=5,tol=1e-3,max_iter=100)
        c, B_OLS_temp = class_SLS.getSLS_Estimator()
        end_time = datetime.datetime.now()
        B_SLS = c * B_OLS_temp
        total_time = end_time - start_time
        total_time= total_time.total_seconds()
        SLS_times.append(float(total_time))
        avg_absErr = np.mean(np.absolute(np.subtract(beta,B_SLS)))
        SLS_absErr.append(avg_absErr)
        errUppBnd = np.add(beta,B_SLS)
        SLS_errUppBnd.append(errUppBnd)
        errLwrBnd = np.subtract(beta,B_SLS)
        SLS_errLwrBnd.append(errLwrBnd)

        SLS_convergence.append(np.absolute(beta - B_SLS) / np.absolute(beta))

        ### citation https://hackernoon.com/gradient-descent-aynk-7cbe95a778da
        ### gradient descent ###
        start_time = datetime.datetime.now()
        W, cost_history, theta_history = gradient_descent(X,y,beta,1e-5,1000)
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        total_time= total_time.total_seconds()
        GD_times.append(float(total_time))
        avg_absErr = np.mean(np.absolute(np.subtract(beta,W)))
        GD_absErr.append(avg_absErr)

        errUppBnd = np.add(beta,W)
        GD_errUppBnd.append(errUppBnd)
        errLwrBnd = np.subtract(beta,W)
        GD_errUppBnd.append(errLwrBnd)


    n_list_plot = [math.log(y,10) for y in n_list]
    plt.clf()
    plt.plot(n_list_plot,SLS_times,'r',label="SLS")
    plt.plot(n_list_plot,GD_times,'b',label="GD")
    plt.title('%s regression times, p=200'%distr)
    plt.legend(loc='upper left')
    plt.ylabel("seconds")
    plt.xlabel("log10(n)")
    #plt.show()
    plt.savefig('%s_regressionTimes.png'%distr)

    plt.clf()
    plt.plot(n_list_plot,SLS_absErr,'r',label="SLS")
    plt.plot(n_list_plot,GD_absErr,'b',label="GD")
    plt.title('%s regression error, p=200'%distr)
    plt.legend(loc='upper left')
    plt.ylabel("avg(|B_pred - B_pop|)")
    plt.xlabel("log10(n)")
    #plt.show()
    plt.savefig('%s_regressionError.png'%distr)




    SLS_convergence = []

    p_list = [10,50,100,200]
    iter = 0
    n = int(1e5)
    for p in p_list:
        iter += 1
        # Generate artificial data
        X = np.random.random((n, 200))
        beta = np.random.rand(200) # dim of feat space

        if distr == "logistic":
            mu, sigma = 0, 1 # mean, std dev
            err = np.random.normal(mu, sigma, n) # normal error
            pr = np.exp(np.dot(X,beta)+err) / (np.add(1,np.exp(np.dot(X,beta)+err)))
            numTrials = 1
            y = np.random.binomial(numTrials, pr, n)

        elif distr == "linear":
            mu, sigma = 0, 10 # mean, std dev
            err = np.random.normal(mu, sigma, n) # normal error

        elif distr == "poisson":
            err = np.random.poisson(10,n)# poisson error, lambda

        y = np.dot(X, beta) + err

        ### scaled least squared estimator ###
        class_SLS = SLS_Estimator(X=X,y=y,genFn=distr,subsetDivisor=5,tol=1e-3,max_iter=1000)
        c, B_OLS_temp = class_SLS.getSLS_Estimator()
        B_SLS = c * B_OLS_temp
        SLS_convergence.append(np.mean(np.absolute(beta - B_SLS) / np.absolute(beta)))

    plt.clf()
    plt.plot(p_list,SLS_convergence,'r',label="SLS")
    plt.title('%s regression convergence, n=1e5'%distr)
    plt.legend(loc='upper left')
    plt.ylabel("avg(|B_pop - B_SLS| / |B_pop|)")
    plt.xlabel("p")
    #plt.show()
    plt.savefig('%s_regressionConvergence.png'%distr)
