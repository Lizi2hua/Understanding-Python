import cv2
import numpy as np

def callback(object):
    pass

img=cv2.imread('images/11.jpg')
# hsv=img
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


# lower_blue=np.array([100,200,100])
# upper_blue=np.array([200,255,200])
cv2.namedWindow('mask')

cv2.createTrackbar('lower[0]','mask',0,255,callback)
cv2.createTrackbar('lower[1]','mask',0,255,callback)
cv2.createTrackbar('lower[2]','mask',0,255,callback)

cv2.createTrackbar('upper[0]','mask',0,255,callback)
cv2.createTrackbar('upper[1]','mask',0,255,callback)
cv2.createTrackbar('upper[2]','mask',0,255,callback)

lower=np.array([100,200,100])
upper=np.array([200,255,200])

while True:
    lower0=cv2.getTrackbarPos('lower[0]','mask')
    lower1=cv2.getTrackbarPos('lower[1]','mask')
    lower2=cv2.getTrackbarPos('lower[2]','mask')
    lower=[lower0,lower1,lower2]
    lower=np.array(lower)

    upper0=cv2.getTrackbarPos('upper[0]','mask')
    upper1=cv2.getTrackbarPos('upper[1]','mask')
    upper2=cv2.getTrackbarPos('upper[2]','mask')
    upper=[upper0,upper1,upper2]
    upper=np.array(upper)

    mask=cv2.inRange(hsv,lower,upper)
    # res=cv2.bitwise_and(img,img,mask=mask)
    print(mask)
    cv2.imshow('origin',img)
    cv2.imshow('hsv',hsv)
    cv2.imshow('mask',mask)
    a=cv2.waitKey(100)
    if  a & 0xff==ord('q'):
        print(a)
        break
cv2.destroyAllWindows()
