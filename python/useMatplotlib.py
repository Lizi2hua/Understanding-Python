# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:45:23 2020

@author: Natuski
"""
import matplotlib.pyplot as plt
import numpy01 as np
#plot(x,y)
#绘制三角函数
#生成三角函数
X=np.linspace(-np.pi,np.pi,256,endpoint=True)
C,S=np.cos(X),np.sin(X)
#print(C,S)
#plt.plot(X,C)
#plt.plot(X,S)
#plt.show()
#test
#python list除法不支持，用np
tp=np.array([22,23,23,22,20,21])
fp=np.array([1,0,0,1,3,2,])
tn=np.array([23,23,22,20,22,21])
fn=np.array([0,0,1,3,1,2])
pre=tp/(tp+fp)
recall=tp/(tp+fn)
plt.plot(recall,pre)
plt.plot(pre,recall)
