import torch
# torch.nn.functional.one_hot
"""torch.nn.functional.one_hot(tensor, num_classes=-1) → LongTensor
Takes LongTensor with index values of shape (*) and returns a tensor of shape (*, num_classes) 
that have zeros everywhere except where the index of last dimension matches the corresponding value of the input tensor, in which case it will be 1."""
a=torch.arange(0,52)
# a必须是整形
# print(a)
onehot=torch.nn.functional.one_hot(a,num_classes=52)
onehot=onehot.tolist()
# 使用索引
A=onehot[0]
print(A)
