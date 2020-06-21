import os
import  torch
from torch.utils import data
from  PIL import Image
import numpy as np
import json
import glob
import torchvision.transforms as transforms

class data_set(data.Dataset):
    # must inherit Dataset class
    def __init__(self,img_path,label_path,transform=None):
        # define path
        # """img_path should be like:path=glob.glob("..\mchar_test_a\*.png")"""
        # tansform参数
        self.img_path=img_path
        self.label_path=label_path
        if transform is not None:
            self.transform=transform
        else:
            self.transform=None


    def __getitem__(self, item):
        # get data
        # item是默认的，用于单个读取时的迭代【个人猜想】
        # data
        imgs=Image.open(self.img_path[item])
        if self.transform is not None:
            imgs=self.transform(imgs)
        # labels
        # 最好传入Dataset类的是已经处理好的label，否则容易报错
        # 5位数（数据集内最高位为6位）定长字符字符串，比如23则是[2,3,10,10,10]
        lbl = np.array(self.label_path[item], dtype=np.int)
        # lbl=np.array(self.label_path[item],dtype=np.int)
        lbl=list(lbl)+(5-len(lbl))*[10]#[2,3]+3*[10]
        return imgs,torch.from_numpy(np.array(lbl[:5]))


    def __len__(self):
        # 定义数据集的大小
        return len(self.img_path)

# 验证数据集


"""定义成功了嘛？"""
if __name__ == '__main__':

    img_path = glob.glob("..\datasets\street_signs\mchar_train\mchar_train\*.png")
    # 最好传入Dataset类的是已经处理好的label，否则容易报错
    label_path = "..\datasets\street_signs\mchar_train.json"
    label_path=json.load(open(label_path))
    label_path=[label_path[x]['label'] for x in label_path]
    print(len(img_path),len(label_path))

    train_loader = torch.utils.data.DataLoader(
        data_set(img_path, label_path,
                 transforms.Compose([
                     transforms.Resize((64, 128)),
                     transforms.RandomCrop((60, 120)),
                     transforms.ColorJitter(0.3, 0.3, 0.2),
                     transforms.RandomRotation(5),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ])),
        batch_size=40,
        shuffle=True,
        num_workers=10,
    )

    for inputs,labels in train_loader:
        print(inputs)
        print(labels)
        break


