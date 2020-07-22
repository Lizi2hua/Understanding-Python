import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

BATCH_SIZE=512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 200


# 可视化输出
def data_viz(train_loader):
    plt.ion()
    for i in range(len(train_loader)):
        imgs = next(iter(train_loader))
        # print(img)
        # print(img[0][0])
        # exit()
        for j in range(train_loader.batch_size):
            img = imgs[0][j]  # [8,1,28,28]，有批次信息和target信息
            label = np.array(imgs[1][j])
            img = np.array(img)
            plt.subplot(4, 4, j + 1)
            plt.imshow(np.transpose(img, (1, 2, 0))[:, :, 0], cmap='gray')
            plt.title(label)
        plt.show()
        plt.pause(0.1)
        plt.clf()


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='datasets', train=True, download=True,  # train=True代表是训练集，False为测试集
                   transform=transforms.Compose([
                       # transforms.RandomRotation(10),
                       transforms.ToTensor(),  # ToTensor要放在图像变换后面，否则就是tensor而不是PIL格式
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)
# data_viz(train_loader)
# exit()

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='datasets', train=False,  # train=True代表是训练集，False为测试集
                   transform=transforms.Compose([
                       # transforms.RandomRotation(10),
                       transforms.ToTensor(),  # ToTensor要放在图像变换后面，否则就是tensor而不是PIL格式
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)


# 定义网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()  # 1,28*28
        self.conv1 = nn.Conv2d(1, 10, 5)  # output:10,24*
        # out=nn.MaxPool2d(F.relu(self.conv1(x)),2,2)
        self.conv2 = nn.Conv2d(10, 20, 3)  # output:20,22*22
        self.fc1 = nn.Linear(20 * 10 * 10, 500)#这儿咋算的？
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        out = F.relu(self.conv2(out))
        #
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)

        return out


# model = SimpleNet().to(DEVICE)
# 迁移学习
model=models.vgg16_bn(pretrained=False)
model.features[0]=nn.Conv2d(1,64,3,padding=5)
model.classifier[6]=nn.Sequential(nn.Linear(4096,10))
print(model)
exit()
# RuntimeError: Given input size: (512x1x1). Calculated output size: (512x0x0). Output size is too small
# https://blog.csdn.net/Zerg_Wang/java/article/details/105249913
model=model.to(DEVICE)
optimizer = optim.Adam(model.parameters())


# 查看模型结构

# for i in range(4):
#     print(next(iter(model.children())))
# print(next(iter(model.modules())))

def train(model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch:{} [{}/{} ({:.0f})]\tLoss:{:.6f}'
                  .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]   #找到概率最大的下标
            correct+=pred.eq(target.view_as(pred)).sum().item()#?

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)