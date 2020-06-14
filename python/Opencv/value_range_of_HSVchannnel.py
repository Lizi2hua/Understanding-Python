import cv2
import  numpy as np
img=cv2.imread('hsv.png')
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h_channel_range=[np.amin(hsv[:,:,0]),np.amax(hsv[:,:,0])]
print(h_channel_range)
s_channel_range=[np.amin(hsv[:,:,1]),np.amax(hsv[:,:,1])]
print(s_channel_range)
v_channel_range=[np.amin(hsv[:,:,2]),np.amax(hsv[:,:,2])]
print(v_channel_range)
