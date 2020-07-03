import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import Dataset

# 生成数据集，y=x[2,-3.4]+4.2
num_inputs=2
num_exampls=1000
true_w=[2,-3.4]
true_b=4.2
# DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x=torch.tensor(np.random.randn(num_exampls,num_inputs))
# x=np.random.randn(num_exampls,num_inputs)
labels=true_w[0]*x[:,0]+true_w[1]*x[:,1]+true_b
# 构建噪声
labels +=np.random.randn()
# print(labels.shape)
# print(x.shape)
plt.subplot(121)
plt.scatter(x[:,0],labels)
plt.subplot(122)
plt.scatter(x[:,1],labels)
plt.show()


# 定义一个函数，每次返回batch_size个随机样本和标签
def data_iter(batch_size,x,labels):
    labels=labels
    batch_size=batch_size
    x=x
    num_exampls_=len(x)
    # 生成索引
    indices=list(range(num_exampls_))
    # 打乱生成的索引，使生成的样本数据顺序随机
    random.shuffle(indices)
    for i in range(0,num_exampls_,batch_size):
        j=torch.tensor(indices[i:min(i+batch_size,num_exampls_)])
        # print(j)
        yield x[j],labels[j]
batch_size=10
# for x ,y in data_iter(batch_size,x,labels):
#     print(x,y)
#     break
# https://zh.gluon.ai/chapter_deep-learning-basics/linear-regression-scratch.html
# 初始化网络参数
# w=np.random.randn(num_inputs,1)
# b=np.zeros(1)
# # print(w,b)
# w=torch.tensor(w,requires_grad=True)
# b=torch.tensor(b,requires_grad=True)

class Line(nn.Module):
    def __init__(self):
        super(Line,self).__init__()
        self.fc=nn.Linear(2,1)

    def forward(self,x):
        out=self.fc(x)
        return out

net=Line()

optimizer=optim.SGD(net.parameters(),lr=0.001)

def train(model,x,labels,optimizer):
    model.train()
    for _,(data,labels) in enumerate(data_iter(batch_size,x,labels)):
        data,labels=data.float(),labels.float()
        optimizer.zero_grad()
        output=model(data)
        #
        criteria=nn.MSELoss()
        loss=criteria(output,labels)

        loss.backward()
        optimizer.step()
        print("loss:{}".format(loss.item()))

train(net,x,labels,optimizer)





