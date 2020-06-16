import cv2
import numpy as np

def callback(object):
    pass

img=cv2.imread("images/2.jpg",0)

# cv2.namedWindow('para')
# cv2.createTrackbar("para",'para',1,255,callback)

kernel=np.array([[1,1,0],[1,0,1],[0,-1,-1]],np.float32)
dst=cv2.filter2D(img,-1,kernel=kernel)
# 均值滤波
dst1=cv2.blur(img,(6,6))
# 高斯滤波
dst2=cv2.GaussianBlur(img,(3,3),1)
dst21=cv2.GaussianBlur(img,(3,3),20)
# 中值滤波
dst3=cv2.medianBlur(img,5)
# 双边滤波
dst4=cv2.bilateralFilter(img,11,75,75)


# 高通滤波
# laplacian锐化
kernel1=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)
dst5=cv2.filter2D(img,-1,kernel=kernel)

# USM算法
dst6=cv2.GaussianBlur(img,(9,9),50)
dst6=cv2.addWeighted(img,1,dst6,-1,0)

# sobel算法
# 不能同时给（1，1）,即x,y同时计算，效果不好
sobel_x=cv2.Sobel(img,-1,1,0,ksize=3)
sobel_y=cv2.Sobel(img,-1,0,1,ksize=3)
sobel=sobel_y+sobel_x
# laplacian算子
dst7=cv2.Laplacian(img,-1)


cv2.imshow('1',img)
# cv2.imshow('x',sobel_x)
# cv2.imshow('y',sobel_y)
cv2.imshow('2',dst7)
cv2.waitKey(150000)

# while True :
#     para=cv2.getTrackbarPos('para','para')
# # cv2.imshow('img',img)
#     dst2=cv2.GaussianBlur(img,(3,3),para)
#     cv2.imshow('dst show',dst2)
#     # cv2.waitKey(2000)