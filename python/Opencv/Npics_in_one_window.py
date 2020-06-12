import cv2
import numpy as np
# 将同一张图片的多个灰度图显示在一个窗口
img=cv2.imread('images/1.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thr=128
ret,binary=cv2.threshold(img,thr,255,cv2.THRESH_BINARY)
ret,binary_inv=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# 截断阀值，大于阀值保留原值，小于阀值归零
ret,trunc=cv2.threshold(img,thr,255,cv2.THRESH_TRUNC)
#
ret,to_zero=cv2.threshold(img,thr,255,cv2.THRESH_TOZERO)
ret,to_zero_inv=cv2.threshold(img,thr,255,cv2.THRESH_TOZERO_INV)

# OTSU二值化
ret,otsu=cv2.threshold(img,0,255,cv2.THRESH_OTSU)

# 装入一个列表之中，好取数据
images=[img,binary,binary_inv,trunc,to_zero,to_zero_inv,otsu]


h,w=np.shape(img)
# 输入几行几列
raws,cols=2,4
stacked_img=np.zeros((h*raws,w*cols))
# 主要目标
# for i in range(raws*cols):
#     j=i+1
#     stacked_img[:h*j,:w*j]=images[i]
# cv2.imshow('stcked pics',stacked_img)
# cv2.waitKey(2000)
# print(thr)

# 正确的代码
stacked_img[:h,:w]=binary
stacked_img[h:h*2,:w]=binary_inv
stacked_img[:h,w:w*2]=trunc
stacked_img[h:h*2,w:w*2]=to_zero
stacked_img[:h,w*2:w*3]=to_zero_inv
stacked_img[h:h*2,w*2:w*3]=otsu
stacked_img[h:h*2,w*3:w*4]=img

cv2.imshow('stcked pics',stacked_img)
cv2.waitKey(-1)