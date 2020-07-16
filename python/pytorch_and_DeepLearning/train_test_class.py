# # # 不是重点，隐藏
import torch, torchvision, os, glob, cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from torch import optim
import time
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 100


# 不重要，隐藏
# 创建自己的数据集
class DigtData(Dataset):
    # 初始化数据集（将数据集读取进来）
    def __init__(self, root, is_train=True):
        self.dataset = []  # 记录所有的数据
        sub_dir = "TRAIN" if is_train else "TEST"  # 根据is_train选择加载的数据集
        for tag in os.listdir(f"{root}/{sub_dir}"):
            img_dir = f"{root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"
                # 封装成数据集
                self.dataset.append((img_path, tag))

    # 统计数据的个数
    def __len__(self):
        return len(self.dataset)

    # 每条数据的处理方式
    def __getitem__(self, index):
        data = self.dataset[index]

        img_data = cv2.imread(data[0], cv2.IMREAD_GRAYSCALE)
        # HWC-->V
        img_data = img_data.reshape(-1)
        # 归一化
        img_data = img_data / 255
        img_data = torch.Tensor(img_data)

        # one-hot
        tag_one_hot = torch.zeros(10)
        tag_one_hot[int(data[1])] = 1

        return img_data, tag_one_hot


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(784, 512, 3),
            nn.Conv1d(512, 128, 3),
        )  # 废弃
        self.fc_block = nn.Sequential(
            nn.Linear(784, 526),
            nn.ReLU(),
            nn.Linear(526, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # out=self.conv_block(x) #NV数据结构不能用卷积
        out = self.fc_block(x)
        return out


root = r'C:\Users\Administrator\Desktop\dataset\MNIST_IMG'


class Train:
    # 训练准备（获取数据）
    def __init__(self, root):
        self.data = DigtData(root=root, is_train=True)
        self.train_data = DataLoader(self.data, batch_size=BATCH_SIZE, shuffle=True,
                                     num_workers=8)  # dataloader是一个迭代器类型，能被for来迭代
        self.val_data = (DigtData(root=root, is_train=False))
        self.val_data = DataLoader(self.val_data, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)
        # 创建模型
        self.net = Net().cuda()
        # 加载参数
        self.net.load_state_dict(
            torch.load(r"C:\Users\Administrator\Desktop\Project：777\CODE\python\pytorch_and_DeepLearning\ckpt\3.t"))
        self.opt = optim.Adam(self.net.parameters())
        #
        self.summary = SummaryWriter('./logs')

    # 训练代码
    def __call__(self):

        for epoch in range(1000):
            self.net.train()
            loss_sum = 0.
            start_time = time.time()
            for i, (x, y_) in enumerate(self.train_data):
                x, y_ = x.cuda(), y_.cuda()
                y = self.net(x)
                # 定义损失
                loss = torch.mean((y - y_) ** 2)

                # 训练三件套
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                loss_sum += loss.detach().item()
            avg_loss = loss_sum / len(self.train_data)
            # print(len(self.train_data))
            end_time = time.time()
            print("epoch:", epoch, "\tLOSS:", avg_loss, "耗时", end_time - start_time)

            with torch.no_grad():
                correct = 0
                start_time = time.time()
                self.net.eval()
                loss_sum_val = 0.
                for i, (x, y_) in enumerate(self.val_data):
                    x, y_ = x.cuda(), y_.cuda()
                    y = self.net(x)
                    # 定义损失
                    loss_val = torch.mean((y - y_) ** 2)
                    loss_sum_val += loss_val.detach().item()
                    # 将网络输出的结果和标签由onehot【0，0，1】变为类别编码【3】
                    pred = torch.argmax(y, dim=1)
                    gt = torch.argmax(y_, dim=1)
                    correct += torch.sum(torch.eq(pred, gt).float())  # 判断对的个数
                test_avg_loss = loss_sum_val / len(self.val_data)
                accuary = correct / len(self.val_data)
                # print(len(self.val_data))
                end_time = time.time()
                print("epoch:", epoch, "\tValidate LOSS:", test_avg_loss, "耗时", end_time - start_time, 'accuary:',
                      accuary.item())
            # 保存网络参数
            torch.save(self.net.state_dict(), f"./ckpt/{epoch}.t")
            self.summary.add_scalars("loss", {"train_loss": avg_loss, "test_loss": test_avg_loss}, epoch)


if __name__ == '__main__':
    train = Train(root)
    train()
    val_data = DataLoader(DigtData(root=root, is_train=False), batch_size=1024, num_workers=3)
    print(val_data)
