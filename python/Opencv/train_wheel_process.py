import cv2

img=cv2.imread('images/train_wheel.jpg')

# 调整亮度，去除纹理，经过调参得alpah=4好些
abs=cv2.convertScaleAbs(img,alpha=4)

# 锐化，提高分辨率
# laplacian=cv2.Laplacian(abs,-1)

guass=cv2.GaussianBlur(abs,(3,3),10)
#开操作,去噪声
# open=cv2.morphologyEx(abs,cv2.MORPH_OPEN,(3,3))

# canny
canny=cv2.Canny(guass,100,200)

# range
# ran=cv2.inRange(abs,120,200)
cv2.imshow('processed',abs)
cv2.imshow('processed1',ran)
cv2.waitKey(0)