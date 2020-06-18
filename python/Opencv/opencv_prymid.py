import cv2
# 高斯金字塔
img=cv2.imread('images/1.jpg')
# for i in range(5):
#     cv2.imshow(f'img{i}',img)
#     # img=cv2.pyrUp(img)
#     img=cv2.pyrDown(img)

# 拉普拉斯金字塔
img_down=cv2.pyrDown(img)
img_up=cv2.pyrUp(img_down)
# 这一步图片变得模糊
cv2.imshow('0',img)
cv2.imshow('1',img_up)
img_new=cv2.subtract(img,img_up)
# 这一步得到边缘图像
cv2.imshow('2',img_new)
cv2.waitKey(0)
