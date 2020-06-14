import cv2
import matplotlib.pyplot as plt

img=cv2.imread("images/1.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# cv2.createTrackbar('threshold value','thresh',0,255)

thr=128
# ret指返回的阈值
# 指定阈值
ret,binary=cv2.threshold(img,thr,255,cv2.THRESH_BINARY)
ret,binary_inv=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# 截断阀值，大于阀值保留原值，小于阀值归零
ret,trunc=cv2.threshold(img,thr,255,cv2.THRESH_BINARY)
#
ret,to_zero=cv2.threshold(img,thr,255,cv2.THRESH_TOZERO)
ret,to_zero_inv=cv2.threshold(img,thr,255,cv2.THRESH_TOZERO_INV)

# OTSU二值化
ret,otsu=cv2.threshold(img,0,255,cv2.THRESH_OTSU)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV','OTSU']

images=[img,binary,binary_inv,trunc,to_zero,to_zero_inv,otsu]

for i in range(7):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i],'gray')
    # cmap='gray'
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()