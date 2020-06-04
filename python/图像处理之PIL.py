#coding:utf-8
from PIL import Image,ImageFilter
import  matplotlib.pyplot as plt 


img=Image.open(r"src/图片3.jpg")
# img.show() 该指令需要系统自带的图片查看器
# 使用matplotlib查看图片不需要自带的图片查看器
# plt.imshow(img)
# plt.title("da da da")
plt.show()

w,h=img.size
print(w,h)
# img=img.resize((4740,3160))
img.thumbnail((w*1.5,h*1.5))
print(img.size)
# 抠图
# img=img.crop((100,100,150,150))
img=img.filter(ImageFilter.CONTOUR)
plt.imshow(img)
plt.title("da da da")
plt.show()