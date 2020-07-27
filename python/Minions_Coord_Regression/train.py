import torch.nn as nn
import torch,os
from torchvision.models import resnet18
from data import dataset
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import numpy as np
from PIL import Image,ImageDraw



DATAPATH=r"C:\Users\Administrator\Desktop\dataset\yellow_coord_regression\labled_pics"
EPOCH=64
BATCH_SIZE=64
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Train():
    def __init__(self):
        self.train_data=DataLoader(dataset(path=DATAPATH),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=0
                                   )

        self.resnet=resnet18()
        self.resnet.fc=nn.Linear(512,4)
        self.model=self.resnet
        self.model.to(DEVICE)
        self.opt=optim.Adam(self.model.parameters())
        ckpt_path="./saved"
        ckpt_file=os.listdir(ckpt_path)
        # print(ckpt_file)
        # exit()
        if len(ckpt_file)>1:
            ckpt_file=os.path.join(ckpt_path,ckpt_file[-1])
            self.model.load_state_dict(torch.load(ckpt_file))


    def __call__(self):
        print("training start!")
        self.model.train()
        for epoch in range(EPOCH):
            loss_sum=0.
            for i ,(input,target) in enumerate(self.train_data):
                input=input.permute(0,3,1,2)
                input,target=input.to(DEVICE),target.to(DEVICE)
                output=self.model(input)
                loss=F.mse_loss(output,target)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                loss_sum+=loss.detach().item() #一个batch的loss
                if i % 10==0:
                    print("Epoch {},batch {},loss:{:.6f}".format(epoch,i,loss.detach().item()))
            avg_loss=loss_sum/len(self.train_data)  #train_data的长度是batch,dataset的长度是整个数据集的长度
            print("\033[1;45m Train Epoch:{}\tavg_Loss:{:.6f} \33[0m".format(epoch,avg_loss))
            torch.save(self.model.state_dict(),f'./saved/{epoch}.t')

            if epoch==EPOCH-1:
                train_data=DataLoader(dataset(DATAPATH),batch_size=16,shuffle=True,num_workers=0)
                for i,(x,y) in enumerate(train_data):
                    x=x.permute(0,3,1,2)
                    imgdata,label=x.to(DEVICE),y.to(DEVICE)
                    out=self.model(imgdata)

                    #画图
                    x=x.permute(0,2,3,1)
                    x.cpu()
                    output=out.cpu().detach().numpy()*300
                    y=y.cpu().numpy()*300

                    img_data=np.array((x[0]+0.5)*255,dtype=np.int8)
                    img=Image.fromarray(img_data,'RGB')
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(output[0],outline="red",width=2)#网络输出的结果
                    draw.rectangle(y[0], outline="yellow",width=2)#原始标签
                    img.show()



if __name__ == '__main__':
  train=Train()
  train()
