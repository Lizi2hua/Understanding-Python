from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open(r"/src/图片01.jpg")
img = np.array(img)
print(img.shape)
plt.imshow(img)
plt.show()

plt.imshow(img.transpose(1, 0, 2))
plt.show()

# reshape参数使用-1是自动计算剩下一个维度
plt.imshow(img.reshape(250, -1, 3))
plt.show()

# imgdata = np.array(np.random.randn(450, 450, 3))
# r,g,b=imgdata[]
# max = max(imgdata)
# min = min(imgdata)
# # 归一化系数
# k = 256 / (max - min)
# imgdata = imgdata(lambda i: k * (i - min))
#
# img_=Image.fromarray(imgdata)
# img_.show()
