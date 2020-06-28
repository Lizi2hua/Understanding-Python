from sklearn import  datasets
import matplotlib.pyplot as plt
import numpy as np

iris=datasets.load_iris()
# print(iris)
iris=iris['data']
# print(iris)
# print("-----------")
sepal_len=iris[:,:1]
# print(sepal_len)
sepal_wid=iris[:,1:2]
petal_len=iris[:,2:3]
petal_wid=iris[:,3:4]
# print(petal_wid)
len=np.append(sepal_len,petal_len)
print(len)