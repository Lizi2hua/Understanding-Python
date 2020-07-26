# 该文件按内容用于测试，不定期删除。“零”没写错！
import os,torch,PIL,glob
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import time
from torch.utils.data import  Dataset
from PIL import  Image
import matplotlib.pyplot as plt
#
# sum_time=[]
#
# for k in range(1,200,1):
#     net=nn.Conv2d(1,1,kernel_size=k)
#     x=torch.randn(1,1,1024,1024)
#     start=time.time()
#     out=net(x)
#     end=time.time()
#     sum_time.append(end-start)
#     print("ksize={},cost time:{}".format(k,sum_time[k-1]))
# plt.xlabel('ksize')
# plt.ylabel('cost time')
# plt.plot([k for k in range(1,200,1)],sum_time)
# plt.show()

# a=torch.randn([64,3,100,100])
# net=nn.Sequential(
#     nn.Conv2d(3,64,3),
#     nn.Flatten(),
#     nn.Linear(64*98*98,10)
# )
# b=net(a)
# print(b.shape)
# print(next(iter(net.modules())))
y=[i for i in range(10)]
train_loss=[i for i in range(0,10,1)]
val_loss=[i for i in range(0,20,2)]
ac=[i for i in range(0,100,10)]

# plt.subplot(121)
# plt.xlabel("epoch")
# plt.ylabel('loss')
# plt.title('the loss change with epcoh')
# plt.plot(y,train_loss,c='r',label="train loss")
# plt.plot(y,val_loss,c='g',label='val loss')
# plt.legend()
# plt.subplot(122)
# plt.xlabel('epoch')
# plt.ylabel('accuarcy')
# plt.title('the accuarcy with epoch')
# plt.plot(y,ac,c='b',label='accuarcy')
# plt.legend()
# plt.show()

acc_all=[]
i=0
start_time=time.time()
for num in range(192,256,8):
    max_acc=torch.rand(1)
    acc_all.append(max_acc)
    i+=1
x_axis=[j for j in range(i)]
plt.title("max acc with differen kernels")
plt.plot(x_axis,acc_all)
plt.show()
