import numpy as np
from sklearn import svm
import  matplotlib.pyplot  as plt
from sklearn import datasets

# 分类问题
iris=datasets.load_iris()
data,target=iris["data"],iris["target"]
# print(data,target)
x=np.array([data[:,0]]).reshape(150,-1)
y=np.array([data[:,1]]).reshape(150,-1)
# print(x)
data=np.concatenate((x,y),axis=1)
target=np.where(target>=1,1,0)
# print(target.shape)
# print(target)
# print(data)
for i in range(len(target)):
    if target[i]==1:
        plt.scatter(x[i],y[i],color='black')
    else:
        plt.scatter(x[i],y[i],color='red')

# plt.show()
model=svm.SVC(C=15,kernel='linear')
# C为惩罚因子，C越大，模型越容易过拟合
model.fit(data,target)
# 获取支持向量
sv=model.support_vectors_
print(sv)
x1=sv[:,0]
y1=sv[:,1]
plt.scatter(x1,y1,color='',marker='o',edgecolors='g',s=100)

# plane=w1*x+b
# plt.plot(x,plane)

plt.show()