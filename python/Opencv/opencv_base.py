import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# read
img=cv2.imread(r"Hydrangea.jpg")
# flags=0为灰度图
cv2.imshow('hydangea',img)
# 非阻塞方法
cv2.waitKey(1000)
# 1000ms
# cv2.waitKey(0)死循环
cv2.destroyWindow(winname='hydangea')
# 强制释放资源，防止资源占用

# wirte
# img_write=np.random.randint(0,255,(300,300,3),np.uint8)
img_write=np.zeros((300,300,3),np.uint8)
# 一张红色的图片BGR
# img_write[:,:,0],img[h,w,c]
img_write[...,2]=255
cv2.imwrite("generate_pic.jpg",img_write)

# 当图片读入opencv时，通道变了
img=cv2.imread(r"Hydrangea.jpg")
img=Image.fromarray(img)
plt.imshow(img)
plt.show()
# 如何变为RGB，切片步长的使用
img=cv2.imread(r"Hydrangea.jpg")
img=img[...,::-1]
# 反着取，即BGR->R->G->B
img=Image.fromarray(img)
plt.imshow(img)
plt.show()




