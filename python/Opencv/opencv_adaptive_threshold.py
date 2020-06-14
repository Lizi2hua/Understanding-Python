import cv2
import matplotlib.pyplot as plt

img=cv2.imread("images/2.jpg",1)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 不加0会报错，为啥？ flags为读取图片的方式,阈值操作在灰度图上，flag=0表示读进来后为灰度图，可以加上面的代码。
ret,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# 2是常数，用来计算计算阈值。阈值=平均值-2.2可以是正负零
th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

title=["origin","binary(v=127)","adaptive mean","adpative guassian"]

imgs=[img,th1,th2,th3]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imgs[i],'gray')
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

