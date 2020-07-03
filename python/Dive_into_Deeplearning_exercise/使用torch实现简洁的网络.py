import torch
import numpy as np
from torch.utils import data
import  matplotlib.pyplot as plt
from torch import nn

ture_w=torch.Tensor([2,-3.4])
ture_b=4.2
n=[2,1000]

# step1 数据处理
def fake_data(true_w,true_b,n):
    x=torch.tensor(np.random.randn(n[1],n[0]))
    labels=ture_w[0]*x[:,0]+ture_w[1]*x[:,1]+ture_b
    labels +=np.random.randn()
    return x,labels.reshape(-1,1)

features,labels=fake_data(ture_w,ture_b,n)

def load_array(data_arrays,batch_size,is_train=True):
    # *data_arrays=data_arrays[0],data_arrays[1],将序列当作位置参数
    # data.TensorDataset:Dataset wrapping tensors.将数据集变成Tensor
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size=10
data_iter=load_array((features,labels),batch_size)

# DataLoader返回的是一个迭代器，我们一般通过迭代器来获取数据(具体访问others/迭代器.py)
print(next(iter(data_iter)))
# print(data_iter)

# step 2: define a network
net=nn.Sequential(nn.Linear(2,1))

# step 3: initial params
# 0 means the first layer
net[0].weight.data.uniform_(0.,0.01)
net[0].bias.data.fill_(0)



