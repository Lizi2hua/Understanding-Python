import torch
import matplotlib.pyplot as plt
from torch import optim
import numpy as np

device = torch.device('cuda')

x_ = torch.linspace(-1, 1, 100).reshape(100, 1)
y =2*torch.sin(x_)+3*torch.cos(x_)
y_ = torch.normal(y, 0.05)


class FCNet(torch.nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, 300)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(300, 300)
        self.fc3 = torch.nn.Linear(300, 1)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.relu(out1)
        out3=self.fc2(out2)
        out4=self.relu(out3)
        y = self.fc3(out4)
        return y


if __name__ == '__main__':
    fc = FCNet()
    opt = optim.Adam(fc.parameters())
    loss = torch.nn.MSELoss()
    for e in range(128):
    # for x, y in zip(x_, y_):
    # for i in range(len(x_)):
    #     x_ = x_[i]
    #     y_ = y_[i]

        z = fc(x_)
        loss1 = loss(y_, z)

        opt.zero_grad()
        loss1.backward()
        opt.step()
        print(loss1)

        v = [fc.forward(j) for j in x_]
        plt.plot(x_, y_,'.')
        plt.plot(x_, v)
        plt.pause(0.1)
        plt.cla()
    plt.show()

