# 该文件按内容用于测试，不定期删除。“零”没写错！
import os,torch,PIL,glob
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import time
from torch.utils.data import  Dataset
from PIL import  Image
import matplotlib.pyplot as plt

sum_time=[]

for k in range(1,200,1):
    net=nn.Conv2d(1,1,kernel_size=k)
    x=torch.randn(1,1,1024,1024)
    start=time.time()
    out=net(x)
    end=time.time()
    sum_time.append(end-start)
    print("ksize={},cost time:{}".format(k,sum_time[k-1]))
plt.xlabel('ksize')
plt.ylabel('cost time')
plt.plot([k for k in range(1,200,1)],sum_time)
plt.show()