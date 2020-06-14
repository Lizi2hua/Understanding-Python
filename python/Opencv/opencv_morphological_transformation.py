import cv2
import matplotlib.pyplot as plt

def callback():
    pass

img=cv2.imread("images/5.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('morphological trans')
cv2.createTrackbar('kernel[0]','morphological trans',1,10,callback)
cv2.createTrackbar('kernel[1]','morphological trans',1,10,callback)




# title=['origin','dilated,max','eroded,min','open','close']
# imgs=[img,dst,dst1,dst2,dst3]

# for i in range(5):
#     plt.subplot(2,3,i+1)
#     plt.title(title[i])
#     plt.imshow(imgs[i],'gray')
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
while True:
    kernel0=cv2.getTrackbarPos('kernel[0]','morphological trans')
    kernel1= cv2.getTrackbarPos('kernel[1]', 'morphological trans')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel0, kernel1))
    # 膨胀、腐蚀都是用核与原图片做卷积运算

    # kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    # kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(6,6))

    dst = cv2.dilate(img, kernel)
    dst1 = cv2.erode(img, kernel)

    # 开操作：先腐蚀再膨胀，去噪
    dst2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # 闭操作：先膨胀再腐蚀，补缺口
    dst3 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # 梯度操作，提轮廓
    dst4=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
    # 礼帽操作
    cv2.imshow('origin',img)
    # cv2.imshow('dilated',dst)
    # cv2.imshow('erode',dst1)
    cv2.imshow('open', dst2)
    cv2.imshow('close', dst3)
    cv2.imshow('gradient',dst4)
    cv2.waitKey(2500)
