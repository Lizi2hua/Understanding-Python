import cv2
import numpy as np
import json
from tool_box import  json_reader


train_json=json.load(open(r"..\datasets\street_signs\mchar_train.json"))

#解决读图的问题

n_pic=30#一次读多少张图


for i in range(n_pic):
    # 第一步:读图
    # 000000怎么自动增加1
    file_name="%06d"%i
    # print(file_name)
    img=cv2.imread(r'..\datasets\street_signs\mchar_train\mchar_train\{}.png'.format(file_name))
    img_arr=json_reader.parse_json(train_json['{}.png'.format(file_name)])
    # cv2.imshow('img',img)
    # cv2.waitKey(100)
    print("slicing pic{}".format(i))
    # 第二部:将数字从图片中切出来
    # 获取数字
    nums=img_arr[4]
    for j in range(len(nums)):
        # 获取框出数字的4个点
        top=img_arr[0,j]
        height=img_arr[1,j]
        left=img_arr[2,j]
        width=img_arr[3,j]
        # print(top)
        # print(height)
        # print(left)
        # print(width)
        # opnecv中图片的维度是[h,w,c]
        img_2=img[top:top+height,left:left+width]
        cv2.imshow('slice',img_2)
        cv2.waitKey(1000)
        cv2.destroyWindow("slice")





# print(img_arr)
# 解决使用定位点将数字切出来的问题
# 第10个图片是1 6
# img_arr的格式是 [top,height,left,width,label]
# img_4=img_arr[4]
# for num in range(len(img_4)):
#
# img=cv2.imread(r'..\datasets\street_signs\mchar_train\mchar_train\000000.png')
# img_arr=json_reader.parse_json(train_json['000000.png'])
# # 1的坐标
# print(img_arr)
# top=img_arr[0,0]
# height=img_arr[1,0]
# left=img_arr[2,0]
# width=img_arr[3,0]
#
# cv2.rectangle(img,(left,top),(left+width,top+height),(0,255,0),1)
# cv2.imshow('number',img)
# cv2.waitKey(0)

