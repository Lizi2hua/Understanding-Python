from  PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#HWC
img=np.array(Image.open('../src/图片01.jpg'))
print(img.shape)

#PIL通道分离
IMG=Image.open('../src/图片01.jpg')
r,g,b=IMG.split()
print(r)
plt.imshow(r)
plt.show()
#点操作
out=IMG.point(lambda i :i*1.5)
plt.imshow(out)
plt.show()
