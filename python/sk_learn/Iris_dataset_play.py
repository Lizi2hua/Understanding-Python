from sklearn import  datasets
import matplotlib.pyplot as plt
import numpy as np
from   sklearn.manifold import TSNE


iris=datasets.load_iris()
# print(iris)
iris_data=iris['data']
label=iris['target']

plt.ion()
for i in range(100):
    tsne1=TSNE(n_components=2,learning_rate=i).fit_transform(iris_data,label)
    # print(tsne)
    plt.scatter(tsne1[:,0],tsne1[:,1],c=label)
    plt.colorbar()
    # if label[i]==0:
    #     plt.scatter(tsne1[:,0],tsne1[:,1],color="r")
    # if label[i]==1:
    #     plt.scatter(tsne1[:,0],tsne1[:,1],color="g")
    # if label[i]==1:
    #     plt.scatter(tsne1[:,0],tsne1[:,1],color="b")
    plt.pause(0.1)
    plt.clf()
plt.ioff()
plt.show()