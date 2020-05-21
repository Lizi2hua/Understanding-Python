import numpy as np
import matplotlib.pyplot as plt
import torch
import random

# label数据
#
# i/100归一化数据
_x = [i / 100 for i in range(100)]
_y = [3 * j + 4 + random.random() for j in _x]
# _y =[j**3 for j in _x]
# 初始化
def train(epoch, lr, _x, _y):
    w = random.random()
    b = random.random()
    s = lr
    # 将数据成对取出
    for i in range(epoch):
        for x, y in zip(_x, _y):
            z = w * x + b
            o = z - y
            loss = o ** 2
            # 求导,先外后内
            dw = -2 * o * x
            db = -2 * o
            # 参数更新
            w = w + s * dw
            b = b + s * db
        if i % 8 == 0:
            plt.cla()
            plt.plot(_x, _y, '.')
            v = [w * e + b for e in _x]
            plt.title("loss:{}".format(loss))
            plt.plot(_x, v)
            plt.pause(0.01)
            print("at epoch:{},w:{},b:{}".format(i, w, b))

    return w, b


plt.ion()
train(epoch=256, lr=0.01, _x=_x, _y=_y)
plt.ioff()
plt.show()
