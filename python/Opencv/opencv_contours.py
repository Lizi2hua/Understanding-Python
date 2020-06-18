import cv2
import numpy as np

img=cv2.imread('images/16.jpg')
print(img.shape)
# img=cv2.Laplacian(img,-1)
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,threshed=cv2.threshold(imggray ,127,255,0)
# cv2.imshow('origin',img)
contours,_=cv2.findContours(threshed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print(contours)
# print(_)

img_c=cv2.drawContours(img,contours,-1,(255,0,0),2)

# 轮廓拟合，最小矩形

print(contours[0])
rect=cv2.minAreaRect(contours[0])
box=cv2.boxPoints(rect)
box=np.int0(box)
img_contour=cv2.drawContours(img,[box],0,(0,0,255),2)

cv2.imshow('2',img_contour)
cv2.imshow('1',img_c)
cv2.waitKey(0)
