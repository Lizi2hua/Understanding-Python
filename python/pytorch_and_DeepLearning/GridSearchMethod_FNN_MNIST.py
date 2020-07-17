import torch,torchvision,os,glob,cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,confusion_matrix,classification_report

# 超参数
DATA_PAHT=r'C:\Users\Administrator\Desktop\dataset\MNIST_IMG'
# TRAIN_PATH=os.path.join(DATA_PAHT,'TRAIN')
# TEST_PATH=os.path.join(DATA_PAHT,'TEST')
# dataset类中定义了is_train,舍弃
EPOCH=200
BATCH_SIZE=526
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集
class DigtData(Dataset):
    def __init__(self,path,is_train,transform=None):
        """
        数据集的路径
        :param path: 数据集的绝对路径
        :param is_train: 根据is_train来打开训练集还是测试集
        :param transform: torchvision.transform
        """
        self._=[]
        self.dataset=[]#数据的绝对路径
        if is_train:
            train_path=os.path.join(path,'TRAIN')
        else:
            train_path=os.path.join(path,'TEST')
        labels=os.listdir(os.path.join(path,train_path))
        for label in labels:
            img_dir=os.path.join(train_path,label)
            img_path=glob.glob(os.path.join(img_dir,'*.jpg'))# 绝对路径
            self._.append(img_path) # 二维数据[[/1/*.jpg][/2//.*jpg]]
        for i in range(len(self._)):
            for j in self._[i]:
                self.dataset.append(j)
        if transform is not None:
            self.transform=transform
        else:
            self.transform=None

    # 如何读一张图片，BATCH由DataLoader管
    def __getitem__(self, idx):
        data=self.dataset[idx]# C:\Users\Administrator\Desktop\dataset\MNIST_IMG\TRAIN\0\1088.jpg
        _=os.path.split(data)# 'C:\\Users\\Administrator\\Desktop\\dataset\\MNIST_IMG\\TRAIN\\0', '1088.jpg'
        label=_[0][-1] # 0

        #读数据
        try:
            img_data=Image.open(data)
        except IOError:
            print("读取数据失败")
        if self.transform is not None:
            img_data=self.transform(img_data)
         # HWC->V,将图片矩阵拉成一个向量
        img_data=img_data.reshape(-1)
        # 归一化,最大最小归一化
        img_data=img_data/255
        #one-hot 编码
        onehot=torch.zeros(10)
        onehot[int(label)]=1

        return img_data,onehot

    def __len__(self):
        return len(self.dataset)

data=DigtData(path=DATA_PAHT,is_train=True,transform=None)
# print(data.__len__())
# exit()

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet,self).__init__()
        self.feature_extractor=nn.Sequential(
            nn.Linear(784,1024),
            nn.ReLU(),
            nn.Linear(1024,2048),
            nn.ReLU(),
            nn.Linear(2048,526),
            nn.ReLU(),
            nn.Linear(526,258)
        )
        self.classifier=nn.Sequential(
            nn.Linear(258,10),
            nn.Softmax(dim=1),
        )
    def forward(self,x):
        out=self.feature_extractor(x)
        out=self.classifier(out)
        return out





def train(train_loader,model,optimizer,epoch):
    # 切换为训练模式
    model.train()

    for batch_idx,(input,target) in enumerate(train_loader):
        # 因为enumerate（）会在前面加上索引
        input,target=input.to(DEVICE),target.to(DEVICE)

        optimizer.zero_grad()
        output=model(input)

        loss=F.mse_loss(output,target)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch:{} [{}/{} ({:.0f})]\tLoss:{:.6f}'
                  .format(epoch, batch_idx * len(input), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

def test(test_loader,model):
    # 切换为训练模式
    model.eval()
    correct=0.
    test_loss_sum=0.


    with torch.no_grad():
        for i,(input,target) in enumerate(test_loader):
            input,target=input.to(DEVICE),target.to(DEVICE)
            output=model(input)
            test_loss=torch.mean((target-output)**2)
            test_loss_sum+=test_loss.item()

            #将onehot转为标签值
            pred=torch.argmax(output,dim=1)
            target=torch.argmax(target,dim=1)
            correct+=torch.sum(torch.eq(pred,target).float())

        accuarcy=correct/len(test_loader)
        print("accuarcy:",accuarcy)
            # pred=torch.argmax(output,dim=1)


#
model=LinearNet().to(DEVICE)


if __name__ =='__main__':
        # 读取数据
    #1.读取训练集的数据
    train_data=torch.utils.data.DataLoader(
        DigtData(path=DATA_PAHT,is_train=True,
                 transform=transforms.Compose(
                     [transforms.RandomRotation(5),
                      transforms.ToTensor()
                     ])),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0)
    print(len(train_data))
    #2.读取测试集的数据
    test_data=torch.utils.data.DataLoader(
        DigtData(path=DATA_PAHT,is_train=True,
                 transform=transforms.Compose(
                     [
                      transforms.ToTensor()
                     ])),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0)
    print(len(test_data))

    model=LinearNet().to(DEVICE)
    optim=torch.optim.Adam(model.parameters())
    for epoch in range (1,EPOCH+1):
        train(train_data,model=model,optimizer=optim,epoch=epoch)
        test(test_data,model=model)











