import torch
import torch.nn.functional as F

input_dir=r"C:\Users\Administrator\Desktop\verifyCode\label\label.txt"

# 生成onehot编码
classes=torch.arange(0,26)
onehot=F.one_hot(classes,num_classes=26)
onehot=onehot.tolist()

# 逐行读取，逐行修改
with open(input_dir,"r") as f:
   tmp=f.readline()
   # 返回为str
   while tmp:
       #  这里写添加onehot的代码

       tmp=f.readline()
