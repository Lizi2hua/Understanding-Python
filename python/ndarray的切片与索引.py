import numpy as np
import torch

a = np.ndarray([1, 2, 3], dtype=np.int8)
# print(type(a))
# print(a.dtype)

"""使用列表生成数组"""
data = [1, 2, 3, 4]
x = np.array(data)
# print(type(x))
# print(x.shape)
# 查看array的维度,就是shape值的长度
# print(x.ndim)
# print("create a empty array:")
# print(np.empty((2,3)))

"""使用arange生成连续的元素"""
# print(np.arange(1,8,2))

a1 = np.arange(20)
a1 = a1.reshape(4, 5)
# print(a1)

"""使用astype赋值数组，并改变元素的数据类型"""
a2 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
b2 = a2.astype(np.int8)
# print(a2,b2)
# 将字符串类型元素转换为数值元素
a3 = np.array(['1', '2', '3', '4'])
b3 = a3.astype(np.int8)
# print(b3)

"""ndarray的运算"""
a4 = np.array([1, 2, 3, 4, 5])
b4 = np.array([2, 1, 3, 6, 3])
# print(a4>b4)


'''========================================'''
'''索引'''
a5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(a5[0][0][0]==a5[0,0,0])
b5 = a5.copy()
c5 = a5
# 生成一个副本
# print((id(c5))==id(a5))
# print(id(b5),id(c5))

'''切片'''
a6 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
# print(a6[:5])

a7 = np.array([[1, 2, 3], [3, 4, 5]])
# 切一次
# print(a7[:1])
# 切两次
# print(a7[:1,:1])
# print(a7[:1][:1])


a8 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
b8 = a8[[0, 1, 2]]
# print(b8)
# print(a8[:2,:1])
'''切片的形状一致才能赋值'''
# a8[:2,:1]=[[7],[8]]
# print(a8)
'''布尔索引'''
c8 = np.array([[[True, True], [False, False]], [[False, False], [False, False]], [[True, True], [True, True]]])
# print(a8[a8<5])
# print(a8[c8])

'''复合索引'''
'''数组索引'''
a8 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
a9 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print("普通索引,a[0]:")
print(a8[0])
print("数组索引,a[[0,1]]:")
print(a8[[0, 1]])
print("数组符合索引,a[[0,1],[0,1]]:")
print(a8[[0, 1], [0, 0]])
print("普通索引+数组索引,a8[[0, 1][1], [0, 1][0],[1]] ")
print(a8[[0, 1][1], [0, 1][0],[1]])

"""索引和切片复合使用"""
print("索引和切片复合使用a8[[0,1][:1]]:")
print(a8[[0,1][:1]])

'''np.where'''
print("np.where")
x=np.array([1,2,3,4,5,6])
x=np.where(x>3,1,0)
print(x)