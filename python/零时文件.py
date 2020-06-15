import cv2
img=cv2.imread('4.jpg')
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dst=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
cv2.imshow('TOPHAT',dst)
cv2.imwrite('Note/src/4_TOPHAT.jpg', dst)
cv2.waitKey(2000)