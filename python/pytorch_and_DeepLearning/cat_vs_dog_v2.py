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




# hyper params
DATAPATH=r"C:\Users\Administrator\Desktop\dataset\cat_dog"
# DATAPATH=r"C:\Users\李梓桦\Desktop\pei_xun\dataset\cat_dog"
EPOCH=500

BATCH_SIZE=16
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 划分数据集
# 图片数据的路径
def ImgLabelGen(root):
    dog_dir=os.path.join(root,'1')
    cat_dir=os.path.join(root,'0')
    img_path=[]
    #dogge 5991
    dog_dir_path=os.listdir(dog_dir)
    for _ in dog_dir_path:
        img_path.append(os.path.join(dog_dir,_))
    #pussy 12000-5991
    cat_dir_path=os.listdir(cat_dir)
    for _ in cat_dir_path:
        img_path.append(os.path.join(cat_dir,_))

    # 生成标签数据
    labels=[]
    for _ in img_path:
        tmp=os.path.split(_)
        label=tmp[1]
        label=label.split(".")
        label=label[0]
        labels.append(label)
    return img_path,labels

def spilted_dataset(root):
    start_time=time.time()

    x,y=ImgLabelGen(root)

    end_time=time.time()
    cost_time=end_time-start_time
    seed=np.ceil(cost_time*100)
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=int(seed))
    return x_train,y_train,x_test,y_test

# img_path,labels,xxx,xxxx=spilted_dataset(root=DATAPATH)
# print(len(img_path))
# print(len(labels))
# img=img_path[0]
# img_data=cv2.imread(img)
#
# img_data=img_data/255
# cv2.imshow('1',img_data)
# cv2.waitKey(0)
# print(img_data)
# print(type(img_data))
# exit()

class DogCat(Dataset):
    def __init__(self,path,is_train,is_fc,transform=None):
        """

        :param path: 数据的绝对路径
        :param is_train: 根据is_train的bool值打开测试集和训练集
        :param transform: 在读取图像时进行图像变换tochvision.transform
        :param is_fc: 如果是全连接网络，需要将数据转成NV结构
        """
        if is_train:
            self.img_path,self.labels,xxx,xxxx=spilted_dataset(root=path)
        else:
            xxx,xxxx,self.img_path,self.labels=spilted_dataset(root=path)

        if transform is not None:
            self.transform=transform
        else:
            self.transform=None

        if is_fc :
            self.fc=True
        else:
            self.fc=False

    def __getitem__(self, idx):
        img=self.img_path[idx]
        label=self.labels[idx]
        img=Image.open(img)
        if self.transform is not None:
            img=self.transform(img)
        #归一化
        if self.fc:
            img=img.reshape(-1)

        #onehot
        onehot=torch.zeros(2)
        onehot[int(label)]=1

        return img,onehot

    def __len__(self):
        return len(self.img_path)

data=DogCat(DATAPATH,is_train=True,is_fc=False,transform=None)
print(data.__len__())
print(data[0])



