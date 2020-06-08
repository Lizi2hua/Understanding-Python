import numpy as np
import torch

# 行列式
a1 = np.array([[1, 2], [3, 4]])
# print(np.linalg.det(a1))
b1 = torch.Tensor([[1, 2], [3, 4]])
# print(b1.det())
# 特殊矩阵
# 对角阵
a2 = np.diag([1, 2, 3, 4])
b2 = torch.diag(torch.Tensor([1, 1, 1, 1]))
# print(a2,b2)
c2 = np.eye(4)  # 单位矩阵
c2_1 = np.eye(3, 4)  # 在计算机中，单位矩阵可以不为方阵
# print(c2)
# print(c2_1)
# 下三角矩阵 上三角矩阵
# 下
a3 = np.tri(3, 3)
b3 = torch.tril(torch.ones(3, 3))
# print(a3)
# print(b3)
# 上 转置 a3.T
# print(a3.T)
# 全零
a4 = torch.zeros(3, 3)
# one hot编码
# 代码求内积
a5 =np.array([1,2])
b5=np.array([3,4])
c5=np.sum(a5*b5)
# print(c5)
# 求特征值
a6=torch.Tensor([[1,2],[3,4]])
# 特征值
print(torch.eig(a6))
# 特征向量
print(torch.eig(a6,eigenvectors=True))