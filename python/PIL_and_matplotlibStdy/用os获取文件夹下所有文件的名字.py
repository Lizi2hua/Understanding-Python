import os
import matplotlib.pyplot as plt
from PIL import Image

path =os.listdir(r"C:\Users\Administrator\Desktop\Project：777\CODE\python\src")
path=path[1:]
# 动态展示图片，一次1秒
plt.ion()
for file in path:
    img=Image.open(r'C:\Users\Administrator\Desktop\Project：777\CODE\python\src/{}'.format(file))
    plt.cla
    plt.imshow(img)
    plt.pause(1)
plt.show()

