import cv2
src=cv2.imread('Hydrangea.jpg')

dst=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)

cv2.imshow('src pic',src)
cv2.imshow('dst show',dst)
cv2.waitKey(1000)