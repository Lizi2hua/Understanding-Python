import cv2

img=cv2.imread('images/car_number.jpg',0)
cv2.imshow('origin',img)

# img=cv2.convertScaleAbs(img,alpha=6,beta=10)
# cv2.imshow('abs',img)

img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,(3,3))
cv2.imshow('close',img)

img=cv2.GaussianBlur(img,(3,3),10)
cv2.imshow('gauss',img)
canny=cv2.Canny(img,100,150)
cv2.imshow('canny',canny)
# img_open=cv2.morphologyEx(canny,cv2.MORPH_OPEN,(3,3))
# cv2.imshow('open',img_open)

cv2.waitKey(0)


