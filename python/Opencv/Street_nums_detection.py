import cv2

img=cv2.imread('images/000268_label353.png')
# img=cv2.imread('images/000995_label74.png')
# img=cv2.imread('images/001928_label38.png')
# 灰度
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# canny
img=cv2.Canny(img,100,200)

cv2.imshow('1',img)
cv2.waitKey(0)