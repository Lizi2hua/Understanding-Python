import cv2

img=cv2.imread("lane.jpg")
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# img=cv2.inRange(img,(25,31,211),(48,14,200))
(thresh, im_bw) = cv2.threshold(hsv[:,:,0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


# cv2.imshow
while True:
    cv2.imshow('pic',img)
    cv2.imshow('hsv',hsv[:,:,:])
    cv2.imshow('otsu', im_bw)
    if  cv2.waitKey(100) & 0xff==ord('q'):
        break
