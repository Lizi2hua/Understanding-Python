import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn.datasets import load_digits
digits = load_digits()

kmeans = KMeans(n_clusters=10, random_state=0)

clusters = kmeans.fit_predict(digits.data)
print(clusters.shape)
#可视化10类中的中心点——最具有代表性的10个数字
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
plt.show()