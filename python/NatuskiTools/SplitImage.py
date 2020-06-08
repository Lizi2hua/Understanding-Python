import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def SplitImage(top_x,top_y,bottom_x,bottom_y,input_dir,name):
    "return a ndarray "
    img=Image.open(input_dir+'/'+name)
    img_arr=np.array(img)

    return img_arr[top_y:bottom_y,top_x:bottom_x]

# input_dir=r"C://Users/李梓桦/Desktop/培训V20200507/CODE/python/NatuskiTools"
#
# img_sliced=SplitImage(top_x=100,top_y=100,bottom_x=200,bottom_y=200,
#                       input_dir=input_dir,name="testpic.jpg")
# plt.imshow(img_sliced)
# plt.show()
#
