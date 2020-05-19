# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:09:53 2020

@author: 李梓桦
theme:梯度下降算法，SGD
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# =============================================================================
# #构造训练数据
# x=np.arange(0.,10.,0.2)
# m=len(x)
# x0=np.full(m,0)#以零填充x0
# inputdata=np.vstack([x0,x]).T #水平堆叠，T转置
# targetdata=2*x+5+np.random.randn(m) #因为是ndarray，所以能这样操作
# =============================================================================


#两种终止条件
loop_max=10000
error=1e-3


