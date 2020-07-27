import os
import  torch
from torch.utils import data
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

import numpy as np

class dataset(data.Dataset):
    def __init__(self,path):
        self.path=path
        self.dataset=[]
        self.dataset.extend(os.listdir(path)) #将path内文件路径全部加载到dataset里面

    def __getitem__(self, idx):
        label=torch.Tensor(np.array(self.dataset[idx].split(".")[1:5],dtype=np.float32)/300) #坐标归一化
        img_path=os.path.join(self.path,self.dataset[idx])
        img=Image.open(img_path)
        img_data=torch.Tensor(np.array(img)/255-0.5) #使数据分布在0的两端，

        return img_data,label

    def __len__(self):
        return len(self.dataset)


# if __name__ == '__main__':
#     DATA_PATH=r"C:\Users\Administrator\Desktop\dataset\yellow_coord_regression\labled_pics"
#     mydata = dataset(DATA_PATH)
#     dataloader = data.DataLoader(dataset=mydata,batch_size=10,shuffle=True)
#     for i,(x,y) in enumerate(dataloader):
#
#
#         print(x.size())
#         print(y)
#         x = x[0].numpy()
#         y = y[0].numpy()
#         # print(y[i]*300)
#         # exit()
#         img_data = np.array((x+0.5)*255,dtype=np.int8)
#
#         img = Image.fromarray(img_data,"RGB")
#         draw = ImageDraw.Draw(img)
#         draw.rectangle(y*300,outline="red",width=2)
#         img.show()

        # fig=plt.figure()
        # ax=fig.add_sublot(111)
        # rect=plt.Rectangle(xy=(y[0:2]*300),((y[3]-y[0])*300),((y[4]-y[1])*300))
        # plt.gca().add_path(plt.Rectangle(xy=y[0:2]*300),(y[3]-y[0])*300,(y[4]-y[1])*300)


















