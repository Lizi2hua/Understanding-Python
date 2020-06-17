import cv2

def callback():
    pass

cv2.namedWindow('para')
img=cv2.imread(r'Opencv\images\train_wheel.jpg',0)
cv2.createTrackbar('thr1','para',0,255,callback)
cv2.createTrackbar('thr2','para',0,255,callback)

abs=cv2.convertScaleAbs(img,alpha=4)
# laplacian=cv2.Laplacian(abs,-1)
# open=cv2.morphologyEx(laplacian,cv2.MORPH_OPEN,(3,3))

while True:
    thr1=cv2.getTrackbarPos('thr1','para')
    thr2=cv2.getTrackbarPos('thr2','para')
    canny=cv2.inRange(abs,thr1,thr2)
    cv2.imshow('b',canny)
    cv2.waitKey(100)
cv2.destroyAllWindows()