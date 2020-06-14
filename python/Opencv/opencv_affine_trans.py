import cv2
import numpy as np

img=cv2.imread("images/1.jpg")
w,h,c=img.shape

affine_Matrix=np.float32([[1,1,0],[0,1,0]])

# 使用函数获得仿射变换矩阵
M=cv2.getRotationMatrix2D((h//2,w//2),45,0.7)

# 注意，是h,w，h指x,w指y
# dst=cv2.warpAffine(img,affine_Matrix,(h,w))
dst=cv2.warpAffine(img,M,(h,w))

cv2.imshow("origin",img)
cv2.imshow("dst",dst)
cv2.waitKey(2500)