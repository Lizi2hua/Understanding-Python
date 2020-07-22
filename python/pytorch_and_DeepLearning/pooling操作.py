import torch.nn as nn
import torch

#自适应池化
#输出为（7x7)
m=nn.AdaptiveAvgPool2d(7)
m2=nn.AdaptiveMaxPool2d((5,4))
input1=torch.randn(1,1,8,9)
input2=torch.randn(1,1,10,10)
print(m(input1).shape)
print(m(input2).shape)