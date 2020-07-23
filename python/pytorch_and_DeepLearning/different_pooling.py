# 使用MNIST验证
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE=128
DEVICE=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
EPOCH=200

#无pooling
class Netv1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, padding=1),  # 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, padding=1),  # 7
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(64 * 7 * 7, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.layers(x)  # NCHW
        # print(h.shape)
        h = h.reshape(-1, 64 * 7 * 7)  # NCHW-->NV
        out = self.output_layer(h)
        return out


class Netv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(64 * 4 * 4, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.layers(x)
        h = h.reshape(-1, 64 * 4 * 4)
        return self.output_layer(h)

# average pooling
class Netv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.AvgPool2d(2),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3),
            nn.AvgPool2d(2),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(64 * 4 * 4, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.layers(x)
        h = h.reshape(-1, 64 * 4 * 4)
        return self.output_layer(h)


class Train():
    def __init__(self):
        self.train_data=torch.utils.data.DataLoader(
            datasets.MNIST(root='datasets',train=True,download=False,
                           transform=transforms.Compose([
                               # transforms.RandomRotation(5),
                               transforms.ToTensor(),
                               # transforms.Normalize((0.1307,),(0.3081))
                           ])),
            batch_size=BATCH_SIZE,shuffle=True
        )
        self.val_data = torch.utils.data.DataLoader(
            datasets.MNIST(root='datasets', train=False, download=True,
                           transform=transforms.Compose([
                               # transforms.RandomRotation(5),
                               transforms.ToTensor(),
                               # transforms.Normalize((0.1307,), (0.3081))
                           ])),
            batch_size=BATCH_SIZE, shuffle=True
        )

        # self.model=Netv1().to(DEVICE)
        self.model = Netv3().to(DEVICE)
        # self.model = Netv1().to(DEVICE)

        ckpt_path="./ckpt"
        ckpt_file = os.listdir(ckpt_path)
        # print(ckpt_file)
        # exit()
        if len(ckpt_file) > 1:
            ckpt_file = os.path.join(ckpt_path, ckpt_file[-1])
            self.model.load_state_dict(torch.load(ckpt_file))
        # self.model.load_state_dict(torch.load(r".\ckpt\.{}t"))
        self.opt = optim.Adam(self.model.parameters(), lr=0.01)
        self.summary = SummaryWriter('./logs')

    def __call__(self):
        for epoch in range(EPOCH):
            self.model.train()
            loss_sum=0.
            start_time=time.time()
            for i,(input,target) in enumerate(self.train_data):
                input,target=input.to(DEVICE),target.to(DEVICE)
                y=self.model(input)
                # print(y.shape)
                loss=F.nll_loss(y,target)


                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                loss_sum+=loss.detach().item()
            avg_loss = loss_sum / len(self.train_data)
            end_time = time.time()
            print('epoch:', epoch, "\tLoss:", avg_loss, "耗时", end_time - start_time)

            with torch.no_grad():
                self.model.eval()
                correct=0
                start_time=time.time()
                loss_sum_val=0

                for i,(input,target) in enumerate(self.val_data):
                    input,target=input.to(DEVICE),target.to(DEVICE)
                    y=self.model(input)

                    loss_val=F.nll_loss(y,target)
                    loss_sum_val+=loss_val.detach().item()

                    pred = torch.argmax(y, dim=1)
                    # print(pred)
                    correct += torch.sum(torch.eq(pred, target).float())
                val_avg_loss = loss_sum_val / len(self.val_data)
                accuracy = correct / (len(self.val_data) * BATCH_SIZE)
                accuracy = accuracy.item()
                end_time = time.time()
                print("epoch:", epoch, "\tValidate LOSS:", val_avg_loss, "耗时", end_time - start_time, 'accuary:',
                      accuracy)

                torch.save(self.model.state_dict(), f'./ckpt/{epoch}.t')
                self.summary.add_scalar('accuracy', accuracy, epoch)
                self.summary.add_scalars('loss', {'train_loss': avg_loss, 'val_loss':val_avg_loss},epoch)



if __name__ == '__main__':
    train = Train()
    train()

