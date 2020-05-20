import torch
from torch import optim
import matplotlib.pyplot as plt

_x = torch.arange(0, 1, 0.01)
_y = 3 * _x + 4 + torch.rand(100)


class Linear(torch.nn.Module):
    def __init__(self):
        # 调用父类得先初始化父类，因为要用父类的方法
        super(Linear, self).__init__()
        # 模型初始化

        self.w = torch.nn.Parameter(torch.rand(1))
        # 定义一个模型的系统参数
        self.b = torch.nn.Parameter(torch.rand(1))

    # 重写forward方法
    def forward(self, x):
        return x * self.w + self.b


if __name__ == '__main__':
    # 实例化
    Linear = Linear()

    # opt = optim.SGD([Linear.w, Linear.b])
    opt = optim.SGD(Linear.parameters(), lr=0.1)

    # 定义损失函数
    loss = torch.nn.MSELoss()
    plt.ion()
    for epoch in range(30):
        for x, y in zip(_x, _y):
            # x通过Linear的构造函数传入forward，因为父类调用了forward，所以可行
            z = Linear(x)
            # 定义损失
            loss = (z - y) **2
            # 梯度清空，必须在求导之前
            opt.zero_grad()
            # 自动求导
            loss.backward()
            # 参数更新
            opt.step()
            print(Linear.w,Linear.b)
            plt.cla()
            plt.plot(_x,_y,".")
            v = [Linear.forward(j) for j in _x]
        plt.plot(_x,v)
        plt.pause(0.01)
    plt.ioff()
    plt.show()
