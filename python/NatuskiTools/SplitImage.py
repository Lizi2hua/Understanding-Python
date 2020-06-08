import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# 外部输入参数
top_x=20
top_y=20
botton_x=200
botton_y=200

# 第一步：读取数据
# 使用绝对路径
input_dir=r"testpic.jpg"
img=Image.open(input_dir)

# 第二步：读取数据的描述性信息
img_arr=np.array(img)
# 转换成numpy后w,h,c变成h,w,c!!!!
plt.imshow(img_arr)
plt.show()
h,w,c=img_arr.shape
print(h,w,c)

# 第三步：切割图像
# 1.使用切片。保留原图片
# 2.将图片切割成可分的任意份输出中间那一份
"w方向切片"
img_arr_scliceW=img_arr[:,:200]
plt.imshow(img_arr_scliceW)
"h放向切片"
img_arr_scliceH=img_arr[:200,:,:]
# img_arr_scliceC=img_arr[:,:,:1] 通道方向切片
plt.imshow(img_arr_scliceH)
plt.show()



