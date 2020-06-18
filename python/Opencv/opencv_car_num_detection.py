import cv2

img_raw=cv2.imread('images/car_num.jpg')
img=img_raw

# 高斯模糊
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sobel算子
# sobelx=cv2.Sobel(img,-1,1,0)
sobely=cv2.Sobel(img,-1,0,1)

# 二值化
ret,img=cv2.threshold(img,0,255,cv2.THRESH_OTSU)
# 开操作
kernelx=cv2.getStructuringElement(cv2.MORPH_RECT,(15,1))
img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernelx)
# 腐蚀
kernely=cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
img=cv2.erode(img,kernely)
# 中值
img=cv2.medianBlur(img,3)

contours,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for item in contours:
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if weight > (height * 2) and 250<weight and height>50:
    # 裁剪区域图片
        chepai =img_raw[y:y + height, x:x + weight]
        cv2.imshow('chepai'+str(x), chepai)

image = cv2.drawContours(img_raw, contours, -1, (0, 0, 255), 3)
# cv2.imshow('1',sobelx)
cv2.imshow('2',img)
cv2.waitKey(0)