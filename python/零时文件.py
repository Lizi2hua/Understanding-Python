# 该文件按内容用于测试，不定期删除。“零”没写错！
import os,torch,PIL,glob
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

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

label=[[0,0,0,0,1,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,1]]
pred=[[0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0]]
print(classification_report(label,pred))
