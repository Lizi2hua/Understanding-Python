import torch
from torch import  nn

if __name__ == '__main__':
    conv=nn.Conv2d(1,1,1,padding=(1,2),bias=False)
    x=torch.randn(1,1,4,4)
    print(conv(x))
    print(conv.weight)
    print(conv.bias)
    print(conv.padding)