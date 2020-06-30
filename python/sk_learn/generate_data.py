import glob
import numpy as np
import os
import matplotlib.pyplot as plt

# 生成数据
# 1.生成回归数据并写
randomstate=np.random.RandomState(9)
x=randomstate.randint(0,100,100)
x=np.array(x).reshape(100,1)
y=x**(1/2)+0.2*np.sin(x)
y=np.array(y).reshape(100,1)
data=np.concatenate((x,y),axis=1)
# plt.scatter(x,y)
# plt.show()
# print(data)

print(x,y)
filename=r"regressionData.txt"
file=open(filename,'w')
for i in range(len(data)):
    str1=str(data[i,0])
    str2=str(data[i,1])
    file.write(str1+' '+str2)
    file.write('\n')





