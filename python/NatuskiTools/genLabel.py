import numpy as np
import os

input_dir = r"C:\Users\Administrator\Desktop\verifyCode"
output_dir = r"C:/Users/Administrator/Desktop/verifyCode/label/"


def text_create(name, input_dir, output_dir, msg):
    txtfile = output_dir + name + '.txt'
    print(txtfile)
    # 创建文件
    file = open(txtfile, 'w')
    for label in msg:
        lable = str(label)
        value = msg.get(lable)
        file.write(label + ' ' + value)
        # 换行
        file.write('\n')


# text_create(name="label", input_dir=input_dir, output_dir=output_dir,msg=msg)


def searchDir(input_dir):
    """检索文件，生成字典，key=文件名字，value=第一个字母"""
    files = os.listdir(input_dir)
    dic = {}
    # 生成的是list文件
    for file in files:
        key = file
        #  获取第一个字母
        value = file[0]
        tmp = {key: value}
        dic.update(tmp)
    return dic


filedict = searchDir(input_dir=input_dir)
text_create(name="label", input_dir=input_dir, output_dir=output_dir, msg=filedict)
