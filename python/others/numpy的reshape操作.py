import torch
import numpy as np

# Tensor默认浮点

a = torch.Tensor([[1, 2], [3, 4]])
b = torch.Tensor([3, 4])
print(a, b)
# 点乘
print(a * b)
# 叉乘
# 如果一个是浮点，一个是整形，不能乘
print(a @ b)
# 转置
print(a.t())
# reshape
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
b = a.reshape(3, 4)
# 四位叉乘
a4 = np.random.rand(3, 3, 3, 4)
b4 = np.random.rand(3, 3, 4, 3)
c = a4 @ b4
print(c, c.shape)
# 换轴操作

a5=np.array([[1,2,3],[2,2,3],[3,2,3]])
b5=a5.transpose(1,0)
print(b5)

a6=torch.rand(1,4,3,5)
print(a6.permute((3,2,1,0)))