import cv2

src=cv2.imread('images/1.jpg')
w,h,c=src.shape

# 放大，由于放大之后有的像素值没有，需要插值，interpolation
dst=cv2.resize(src,(w*3,h*3),interpolation=cv2.INTER_CUBIC)
# 转置,逆时针旋转
dst2=cv2.transpose(src)
# 反转 0上下反转，大于0左右翻转，负数上下左右翻转
dst3=cv2.flip(src,-1)
cv2.imshow("resize",dst)
cv2.imshow("transpose",dst2)
cv2.imshow("flip",dst3)
cv2.waitKey(2000)