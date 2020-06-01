# from sklearn.datasets import load_boston
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy

# 生成数据
x = np.array([i for i in range(0, 100)],dtype=np.float)
x=torch.from_numpy(x)
x=x.reshape(100,1)
y = torch.sin(x / 100) ** 2 + 0.3 * torch.cos(x / 50) ** 3 + 4.3 * torch.sin(x / 15) ** 4

# plt.plot(x,y,'.')
# plt.show()

# 定义模型
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.w1 = nn.Parameter(torch.randn(1, 80))
        self.b1 = nn.Parameter(torch.zeros(80))

        self.w2 = nn.Parameter(torch.randn(80, 160))
        self.b2 = nn.Parameter(torch.zeros(160))

        self.w3 = nn.Parameter(torch.randn(160, 80))
        self.b3 = nn.Parameter(torch.zeros(80))

        self.w4 = nn.Parameter(torch.randn(80, 40))
        self.b4 = nn.Parameter(torch.zeros(40))

        self.w5 = nn.Parameter(torch.randn(40, 1))
        self.b5 = nn.Parameter(torch.randn(1))

    def forward(self, x):
        fc1 = F.relu(torch.matmul(x, self.w1) + self.b1)
        fc2 = F.relu(torch.matmul(fc1, self.w2) + self.b2)
        fc3 = F.relu(torch.matmul(fc2, self.w3) + self.b3)
        fc4 = F.relu(torch.matmul(fc3, self.w4) + self.b4)
        fc5 = torch.matmul(fc4, self.w5) + self.b5
        return fc5


# if "__name__" == "__main__":
net = FCNet()
optim = torch.optim.Adam(net.parameters(),lr=0.1)
loss_func = nn.MSELoss()
plt.ion()
for i in range(10000):
    out = net(x.float())
    loss = loss_func(out, y.float())

    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 10 == 0:
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, out.detach().numpy(), 'r')
        plt.pause(0.1)
        print(loss.item())
plt.ioff()
plt.show()
