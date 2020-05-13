# https://docs.python.org/zh-cn/3/howto/argparse.html
# ***********************************#
import argparse
# 1.声明一个parser
parser = argparse.ArgumentParser()
# ***********************************#
# 2.添加一个参数
parser.add_argument("path",help="文件的路径",default="请输入文件路径") #位置参数，表示第一个出现的参数赋值给parg
# 3.读取命令行参数
args=parser.parse_args()
# 4.调用这些参数
print(args.path)
