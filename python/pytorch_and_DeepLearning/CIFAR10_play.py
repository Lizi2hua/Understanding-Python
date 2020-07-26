import torch,torchvision,time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets
from torchvision import models
import matplotlib.pyplot as plt
from torch import optim

BATCH_SIZE=128
EPOCH=50
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root=r'C:\Users\Administrator\Desktop\dataset\CIFAR10'


class ConvNet(nn.Module):
    def __init__(self,layer1_filter):
        super(ConvNet, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,layer1_filter,3,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),

            nn.Conv2d(layer1_filter,384,3,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),

            nn.Conv2d(384,384,3),
            nn.ReLU(),
            nn.Conv2d(384,256,3),
            nn.ReLU(),

            # nn.Conv2d(256,1,1),
            # nn.ReLU()
        )
        self.classifier=nn.Sequential(
            nn.Linear(256*4*4,4096),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,10),
            nn.Dropout(),
            nn.Softmax(dim=1),
        )
    def forward(self,x):
        out=self.features(x)
        # print(out.shape)
        out=out.reshape(-1,256*4*4)
        out=self.classifier(out)
        return out


class Train():
    def __init__(self,root,filter_nums):
        self.train_data=DataLoader(datasets.CIFAR10(root=root,train=True,download=True,transform=transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(0.1,0.1,0.1,[-0.5,0.2]),
                transforms.ToTensor()
            ])),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=10)
        self.val_data=DataLoader(datasets.CIFAR10(root=root,train=False,download=True,transform=transforms.Compose(
            [
                transforms.ToTensor()
            ])),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=10)
        self.filter_nums=filter_nums
        self.model=ConvNet(layer1_filter=filter_nums).to(DEVICE)
        self.opt=optim.Adam(self.model.parameters())

    def __call__(self):
        train_loss=[]
        val_loss=[]
        ac=[]
        max_acc=0.
        print("training start!")
        self.model.train()
        for epoch in range(EPOCH):
                loss_sum=0.
                for i ,(input,target) in enumerate(self.train_data):
                    #将target变为onehot编码
                    target=F.one_hot(target,num_classes=10).float()
                    input,target=input.to(DEVICE),target.to(DEVICE)
                    output=self.model(input)
                    # print('output',output)
                    # print(output.shape)
                    # print('target',target)
                    # print(target.shape)
                    # exit()
                    loss=F.mse_loss(output,target)
                    # print('loss',loss)
                    # print(loss.shape)
                    # exit()

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    loss_sum+=loss.detach().item() #一个batch的loss
                    # if i % 100==0:
                    #     print("Epoch {},batch {},loss:{:.6f}".format(epoch,i,loss.detach().item()))
                avg_loss=loss_sum/len(self.train_data)  #train_data的长度是batch,dataset的长度是整个数据集的长度
                train_loss.append(avg_loss)
                if epoch%10==0:
                  print("\033[1;45m Train Epoch:{}\tavg_Loss:{:.6f} \33[0m".format(epoch,avg_loss))

                correct=0.
                loss_sum_val=0.
                self.model.eval()
                for i,(input,target) in enumerate(self.val_data):
                    #将target变为onehot编码
                    target=F.one_hot(target,num_classes=10).float()
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
                #统计整个学习周期中最大的精度值
                if accuarcy>max_acc:
                    max_acc=accuarcy
                val_loss.append(val_avg_loss)
                ac.append(accuarcy)
                if epoch%10==0:
                  print("\033[1;45m Train Epoch:{}\tavg_val_Loss:{:.6f},correct:{}/{},accuarcy:{} \33[0m".format(epoch, val_avg_loss,correct,len(self.val_data)*BATCH_SIZE,accuarcy))

        return max_acc
        # y=[i for i in range(EPOCH)]
        # plt.subplot(121)
        # plt.xlabel("epoch")
        # plt.ylabel('loss')
        # plt.title('the loss change with epcoh')
        # plt.plot(y,train_loss,c='r',label="train loss")
        # plt.plot(y,val_loss,c='g',label='val loss')
        # plt.legend()
        # plt.subplot(122)
        # plt.xlabel('epoch')
        # plt.ylabel('accuarcy')
        # plt.title('the accuarcy with epoch,kernels={}'.format(self.filter_nums))
        # plt.plot(y,ac,c='b',label='accuarcy')
        # plt.legend()
        # plt.show()


if __name__ == '__main__':
    acc_all=[]
    i=0
    start_time=time.time()
    for num in range(192,256,8):
      start_time=time.time()
      train = Train(root,filter_nums=num)
      max_acc=train()
      print(max_acc)
      acc_all.append((max_acc))
      i+=1
      end_time=time.time
      print('finish train with first kerness {}'.format(num))
    end_time=time.time()
    print('finish training,cost {} seconds'.format(end_time-start_time))
    x_axis=[j for j in range(i)]
    plt.title("max acc with differen kernels")
    plt.plot(x_axis,acc_all)
    plt.show()




#
# if __name__ == '__main__':
#     # model=models.alexnet(pretrained=False)
#     # # model.classifier=nn.Sequential()
#     # print(next(iter(model.modules())))
#     a=torch.rand([64,3,32,32])
#     net=ConvNet()
#     b=net(a)
#     print(b.shape)