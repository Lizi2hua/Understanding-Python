import numpy as np
# 对一维数据的填充
arr_1=np.array([1,1,1,1])
arr_1_padded=np.pad(arr_1,1,constant_values=0)
# 第二个参数表示填充几次
print(arr_1_padded)
# 对二维数据的填充
arr_2=np.array([[1,1],[1,1]])
arr_2_padded=np.pad(arr_2,1,constant_values=0)
print(arr_2_padded)
