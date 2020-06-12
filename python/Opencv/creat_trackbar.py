import cv2
import numpy as np


# 定义回调函数，此程序无需回调，所以Pass即可
def callback(object):
    pass


img = np.zeros((500, 400, 3), np.uint8)
cv2.namedWindow('image')

# 创建一个开关滑动条，只有两个值，起开关按钮作用
switch = '0:OFF\n1:ON'
cv2.createTrackbar(switch, 'image', 0, 1, callback)

cv2.createTrackbar('R', 'image', 0, 255, callback)
cv2.createTrackbar('B', 'image', 0, 255, callback)
cv2.createTrackbar('G', 'image', 0, 255, callback)

while(True):
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    if cv2.getTrackbarPos(switch, 'image') == 1:
        img[:] = [b, g, r]
    else:
        img[:] = [255, 255, 255]
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    cv2.imshow('image', img)
cv2.destroyAllWindows()
