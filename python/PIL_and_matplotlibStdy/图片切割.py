import numpy as np
from PIL import Image

img = Image.open(r"图片3.jpg")
img_data = np.array(img)
w, h, c = img_data.shape
"""重点：这样切才能保证数据不变"""
# 如果（2，2，w//2,h//2,c）咋样

img_data = img_data.reshape(2, w // 2, 2, h // 2, c)
# 换轴
img_data=img_data.transpose(0,2,1,3,4)

img_data=img_data.reshape(-1,w//2,h//2,3)

img_data=np.split(img_data,4,axis=0)

for i,img_d in enumerate(img_data):
    img_d = img_d[0]
    img=Image.fromarray(img_d,mode='RGB')
    img.save("pic{}.jpg".format(i))