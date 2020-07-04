import cv2
import matplotlib.pyplot as plt

img=cv2.imread('images/car_num.jpg')
# 计算B通道的直方图
hist=cv2.calcHist([img],[0],None,[256],[0,256])

plt.plot(hist)
plt.show()

# 直方图均衡化
# 1.全局均衡化
img_1=cv2.imread('images/12.jpg',0)
dst=cv2.equalizeHist(img_1)
# 2.局部均匀化
dst2=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8))
dst2=dst2.apply(img_1)
cv2.imshow('equalize1',dst)
cv2.imshow('equalize2',dst2)
cv2.waitKey(0)