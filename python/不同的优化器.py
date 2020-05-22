import torch
from torch import optim
import matplotlib.pyplot as plt

_x = torch.arange(0, 1, 0.01)
_y = 3 * _x + 4 + torch.rand(100)


class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.w = torch.nn.Parameter(torch.rand(1))
        self.b = torch.nn.Parameter(torch.rand(1))

    def forward(self, x):
        return self.w * x + self.b


if __name__ == "__main__":
    Line = Linear()
    epoch = 64
    opt = optim.Adam(Line.parameters())
    # opt = optim.SGD(Line.parameters(), lr=0.1)

    plt.ion()
    for index in range(epoch):
        for x,y in zip(_x,_y):
            z=Line.forward(x)

            loss=(z-y)**2

            opt.zero_grad()

            loss.backward()

            opt.step()

        plt.cla()
        plt.plot(_x,_y,'.')
        v=[Line.forward(j) for j in _x]
        plt.plot(_x,v)
        plt.title("epoch:{} loss:{}".format(index,loss))
        plt.pause(0.01)

    plt.ioff()
    plt.show()






