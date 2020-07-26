from data import DogCat
from net import ConvNet
import torch,os,time
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch import optim

DATAPATH=r"C:\Users\Administrator\Desktop\dataset\cat_dog"
# DATAPATH=r"C:\Users\李梓桦\Desktop\pei_xun\dataset\cat_dog"
EPOCH=2
BATCH_SIZE=64
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Train():
    def __init__(self,root,filter_nums):
        self.train_data=DataLoader(DogCat(path=root,is_train=True,transforms=transforms.Compose(
            [transforms.RandomRotation(10),
             transforms.ToTensor()]
        )),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=0
                                   )
        self.val_data=DataLoader(DogCat(path=root,is_train=False,transforms=transforms.ToTensor()),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=0
                                   )

        self.model=ConvNet(layer1_filter=filter_nums).to(DEVICE)
        self.opt=optim.Adam(self.model.parameters())
        self.summary=SummaryWriter('./logs')
        # 保存模型


    def __call__(self):
        print("training start!")
        self.model.train()
        for epoch in range(EPOCH):
            loss_sum=0.
            for i ,(input,target) in enumerate(self.train_data):
                input,target=input.to(DEVICE),target.to(DEVICE)
                output=self.model(input)
                # print('output',output)
                # print(output.shape)
                # print('target',target)
                # print(target.shape)
                loss=F.mse_loss(output,target)
                # print('loss',loss)
                # print(loss.shape)
                # exit()

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                loss_sum+=loss.detach().item() #一个batch的loss
                if i % 100==0:
                    print("Epoch {},batch {},loss:{:.6f}".format(epoch,i,loss.detach().item()))
            avg_loss=loss_sum/len(self.train_data)  #train_data的长度是batch,dataset的长度是整个数据集的长度
            print("\033[1;45m Train Epoch:{}\tavg_Loss:{:.6f} \33[0m".format(epoch,avg_loss))

            correct=0.
            loss_sum_val=0.
            self.model.eval()
            for i,(input,target) in enumerate(self.val_data):
                input,target=input.to(DEVICE),target.to(DEVICE)
                output=self.model(input)

                loss_val=F.mse_loss(output,target)
                loss_sum_val+=loss_val.detach().item()

                #torch.argmax返回维度最大的索引
                pred=torch.argmax(output,dim=1) #预测的类，[[1],[0],...]这样的数据类型
                #将target的onehot编码变为类名
                cls=torch.argmax(target,dim=1)
                # 计算预测对的个数
                correct+=torch.sum(torch.eq(pred,cls).float())
            val_avg_loss=loss_sum_val/len(self.val_data)
            accuarcy=correct/(len(self.val_data)*BATCH_SIZE)
            accuarcy=accuarcy.item()
            print("\033[1;45m Train Epoch:{}\tavg_val_Loss:{:.6f},correct:{}/{},accuarcy:{} \33[0m".format(epoch, val_avg_loss,correct,len(self.val_data)*BATCH_SIZE,accuarcy))
            self.summary.add_scalar('accuarcy',accuarcy,epoch)
            self.summary.add_scalars('loss',{'train_loss':avg_loss,'val_loss':val_avg_loss},epoch)

if __name__ == '__main__':
    for i in range(8,256,2):
        train = Train(DATAPATH,filter_nums=i)
        train()
