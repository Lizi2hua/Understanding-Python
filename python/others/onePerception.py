# -*- coding: utf-8 -*-
"""
Created on Mon May 18 22:03:15 2020

@author: 李梓桦
theme:单感受器
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

batch = 32
lr = 0.0001
n = 128


def sgn(x):
    if x > 0:
        return 1
    else:
        return -1


def dataSet(n):
    x1_ = np.array([i / n for i in range(n)])
    x2_ = np.random.rand(n, )
    y_ = [1 if (x2_[i] + 0.02) > x1_[i] else -1 for i in range(n)]
    return x1_, x2_, y_


def showData(x1_, x2_, y_):
    for i in range(len(y_)):
        if y_[i] == 1:
            plt.scatter(x1_[i], x2_[i], color="red")
        if y_[i] == -1:
            plt.scatter(x1_[i], x2_[i], color="black")
    plt.show()


def crossEntropy(t, p):
    CE =-(p*np.log(t)+(1-p)*np.log(1-t))
    return CE

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

x1_, x2_, y_ = dataSet(n)
# showData(x1_,x2_,y_)
for i in range(batch):
    for x1, x2, y in zip(x1_, x2_, y_):
        pre = sgn(x1 * w1 + x2 * w2 + b)

        # 得用交叉熵
        loss = crossEntropy(y_,pre)

        # 求负梯度
        # 权值更新策略
        dw1 = -2 * o * x1
        dw2 = -2 * o * x2
        db = -2 * o

        # 梯度下降
        w1 = w1 + lr * o * x1
        w2 = w1 + lr * o * x2
        b = b + lr * o * b

        print(w1, w2, loss)
# test
for i in range(len(y_)):
    if y_[i] == 1:
        plt.scatter(x1_[i], x2_[i], color="red")
    if y_[i] == -1:
        plt.scatter(x1_[i], x2_[i], color="black")

xt = [i / 20 for i in range(20)]
x2t = [(b - w1 * xt[i]) / w2 for i in range(20)]
label = [sgn(xt[i] * w1 + xt[i] * w2 + b) for i in range(len(xt))]

plt.plot(xt, x2t)
plt.show()
