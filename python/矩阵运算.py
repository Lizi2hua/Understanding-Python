import numpy as np
import torch

a=np.array([[1,2],[3,4]])
b=np.array([[1,2],[3,4]])
a1=torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
a2=torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
a3=torch.from_numpy(a)
#加减
c=a+b
# print(c,type(c))

#数加
d=4+a
# print(d,type(d))

#叉乘
e=np.dot(a,b)
e1=torch.matmul(a1,a2)
# print(e1,e1.shape)
#点乘
f=a*b
# print(f)

#转置
print(a3.T)
# print(a3.t())