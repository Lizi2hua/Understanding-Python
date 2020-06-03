from PIL import Image
import  matplotlib.pyplot as plt 


img=Image.open("src/图片3.jpg")
# img.show() 该指令需要系统自带的图片查看器
# 使用matplotlib查看图片不需要自带的图片查看器
plt.imshow(img)
plt.show()
