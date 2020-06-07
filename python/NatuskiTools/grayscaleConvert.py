import numpy as np
from PIL import Image
import os

# w,h,c=60,60,3

out_dir = (r"C:/Users/Administrator/Desktop/verifyCode/grayscale/")


# 批量转换成灰度图片
def convert():
    """使用后记得删除生成的文件"""
    file = os.listdir(r"C:\Users\Administrator\Desktop\verifyCode")
    out_dir = (r"C:/Users/Administrator/Desktop/verifyCode/grayscale/")
    os.mkdir(out_dir)
    print("making dir and begining to convert!")
    for num in file:
        img = Image.open(r"C:/Users/Administrator/Desktop/verifyCode/" + num).convert("L")
        print(out_dir + num)
        img.save(out_dir + num)
    print("done!")

# os.remove(r"C:/Users/Administrator/Desktop/verifyCode/grayscale")这一行需要权限，不remove的话使用前删除
# convert()
