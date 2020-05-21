from  PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img=Image.open(r"C:\Users\Administrator\Desktop\Project：777\CODE\python\src\图片01.jpg")
img=np.array(img)
print(img.shape)
plt.imshow(img)
plt.show()