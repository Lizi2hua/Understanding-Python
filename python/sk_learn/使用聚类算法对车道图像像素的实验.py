# import sklearn
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler
# from PIL import Image
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import cv2
#
# # 读图
# img_path=r"imgs\lane.jpg"
# img=cv2.imread(img_path)
#
#
# hist_R=cv2.calcHist([img],[0],None,[256],[0,256])
# hist_G=cv2.calcHist([img],[1],None,[256],[0,256])
# hist_B=cv2.calcHist([img],[2],None,[256],[0,256])
# # 特征分析
# # plt.subplot(131)
# # plt.plot(hist_R)
# # plt.subplot(132)
# # plt.plot(hist_G)
# # plt.subplot(133)
# # plt.plot(hist_B)
# # plt.show()
# # plt.subplot(131)
# # plt.scatter(img[:,:,0][0],img[:,:,0][1])
# # plt.subplot(132)
# # plt.scatter(img[:,:,1][0],img[:,:,1][1])
# # plt.subplot(133)
# # plt.scatter(img[:,:,2][0],img[:,:,2][1])
# # plt.show()
# # 归一化
# R=img[0:600,0:900,0]
# G=img[0:600,0:900,1]
# B=img[0:600,0:900,2]
#
# # scaler=MinMaxScaler()
# # r=scaler.fit_transform(R)
# X=R
#
# result=y_pred = KMeans(n_clusters=10, random_state=None).fit_predict(X)
# # print(result.shape)
# # exit()
# plt.scatter(X[:,0],X[:1],c=result)
# plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度颜色
img = cv2.imread('imgs\lane4.jpg', 0)
print(img.shape)

#获取图像高度、宽度
rows, cols = img.shape[:]

#图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

#定义中心 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, flags)

#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))
# print(dst)
# exit()

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()
