# -*- coding: utf-8 -*-
"""
Created on Mon May 18 22:03:15 2020

@author: 李梓桦
theme:基本初等函数
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

# 创建绘图区域
plt.figure("函数图像")

# *****************************#
# 0.numpy和tensor的数据类型
a = torch.tensor(1)
print("a's type:", type(a))
b = np.array(1)
print("b's type:", type(b))
# tensro 和 ndarray的相互转换
c = a.numpy()
# 错误方法
# d=b.tensor()
d = torch.from_numpy(b)
print("a change into ndarray:", type(c))
print("b change into tnesor:", type(d))
# *****************************#
# 1.常函数(利用numpy的广播机制)
x0 = np.arange(-10, 10)
y0 = np.tile(np.array([3]), x0.shape)
plt.subplot(331)
plt.plot(x0, y0)
plt.show
# 2.幂函数
a = 2
x = np.arange(-10, 10, 0.1)
y = x ** 2
# plot:折线
plt.subplot(332)
plt.plot(x, y)
plt.show()
# 3.指数函数
a = 3
x3 = np.arange(-3, 3, 0.1)
y3 = a ** x3
plt.subplot(333)
plt.plot(x3, y3)
plt.show

a2 = 0.5
x4 = np.arange(-3, 3, 0.1)
y4 = a2 ** x4
plt.plot(x4, y4)
plt.show
# 4.对数函数
x5 = np.arange(0.1, 10, 0.1)
y5 = np.log(x5)
yz = np.zeros_like(x5)
plt.plot(x5, y5)
plt.show
plt.plot(x5, yz)
plt.show
# 5自定义底数：换底公式
x6 = np.arange(0.1, 10, 0.1)
y6 = np.log(x6) / np.log(0.1)
plt.plot(x6, y6)
plt.show
# 6sigmoid
x7 = np.arange(-10, 10, 0.1)
y7 = 1 / (np.exp(-x7))
plt.subplot(1, 2, 1)
plt.plot(x7, y7)
# tanh
