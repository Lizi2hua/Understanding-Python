# 该文件按内容用于测试，不定期删除。“零”没写错！
import os,torch,PIL,glob
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import time
from torch.utils.data import  Dataset
from PIL import  Image
# DATA_PAHT=r'C:\Users\Administrator\Desktop\dataset\MNIST_IMG'
# train_path=os.path.join(DATA_PAHT,'TRAIN')
# labels=os.listdir(train_path) #labels即是标签，也是子文件名称
# print(labels)
# _=[]
# for label in labels:
#     img_dir=os.path.join(train_path,label)
#     img_path=glob.glob(os.path.join(img_dir,'*.jpg'),)#输出是绝对路径
#     # img_path=os.listdir(img_dir)#输出是相对路径
#     print(len(img_path))
#     _.append(img_path)
# # print(_)
# dataset=[]
# for i in range(len(_)):
#     for j in _[i]:
#         # print(j)
#         dataset.append(j)
# data=dataset[100]
# print(data)
# path=os.path.split(data)
# print(path)
# print(path[0][-1])
# onehot=torch.zeros(10)
# onehot[2]=1
# print(onehot)
#
# exit()
#
# class Mynet(nn.Module):
#     def __init__(self):
#         super(Mynet,self).__init__()
#         self.conv_block=nn.Sequential(
#             nn.Conv2d(784,526,3),
#             nn.ReLU(),
#             nn.Conv2d(526,258,3),
#             nn.ReLU(),
#             nn.Conv2d(258,124,3),
#             nn.ReLU()
#         )
#         self.dense_block=nn.Sequential(
#             nn.Linear(128,128),
#             nn.ReLU(),
#             nn.Linear(128,10),
#             nn.Softmax()
#         )
#     def forward(self,x):
#         out=self.conv_block(x)
#         out=self.dense_block(x)
#         return out
#
#
# class Mynet2(nn.Module):
#     def __init__(self):
#         super(Mynet2,self).__init__()
#         self.conv_layer1=nn.Conv2d(128,128,3)
#         self.conv_layer2=nn.Conv2d(128,128,3)
#         self.conv_layer3=nn.Conv2d(128,128,3)
#         self.fc_layer1=nn.Linear(128,128)
#         self.fc_layer2=nn.Linear(128,10)
#     def forward(self,x):
#         out=self.conv_layer1(x)
#         out=nn.functional.relu(out)
#
#         out=self.conv_layer2(out)
#         out=nn.functional.relu(out)
#
#         out=self.conv_layer3(out)
#         out=nn.functional.relu(out)
#
#         out=self.fc_layer1(out)
#         out=nn.functional.relu(out)
#
#         out=self.fc_layer2(out)
#         out=nn.functional.softmax(out)
#         return out
#
# model=Mynet2()
# print(model)

# a=torch.Tensor([[0.9,0.1,0.3,0.8,0.9,0.2],[0.9,0.1,0.3,0.8,0.9,0.2],[0.9,0.1,0.3,0.8,0.9,0.2]])
# expa=torch.exp(a)
# print(expa)
# sum=torch.sum(expa,dim=1,keepdim=True)
# print(sum)
# print(expa/sum)
# print(torch.nn.Softmax())
# start_time=time.time()
# for i in range(100000):
#     a=1+2
# end_time=time.time()
# print(end_time-start_time,'s')
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
        print(len(self.dataset))
        return len(self.dataset)

