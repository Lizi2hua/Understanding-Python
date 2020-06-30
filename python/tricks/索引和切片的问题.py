import numpy as np

# 本质是索引
a=np.array([[3,5,9],[4,2,1],[6,3,7]])
print(a[[2,0,1],[1,2,0]])
# 切片加索引
x=np.arange(12).reshape(3,2,2)
print(x[:2,1:,0])