import os
import argparse

#
default_path = os.getcwd()
parser = argparse.ArgumentParser()
#
parser.add_argument("--path", default=default_path)  # 位置参数，表示第一个出现的参数赋值给parg
#
args = parser.parse_args()
path = args.path
files = os.listdir(path)
#
line_code = 0
line_space = 0
line_annotate = 0
line_num = 0
#
print("读取至以下文件：")
file_num = 1
for file in files:
    if not os.path.isdir(file):
        f = open(path + "/" + file, mode="r", encoding="UTF-8")
        try:
            for line in f:
                if len(line) == 1:
                    line_space += 1
                elif line[0] == "'" or line[0] == "#" or line[0] == '"':
                    line_annotate += 1
                elif line[0] == " ":
                    line = line.strip()
                    if len(line) == 0:
                        line_space += 1
                    if line[0] == "'" or line[0] == "#" or line[0] == '"':
                        line_annotate += 1
                    else:
                        line_code += 1
                else:
                    line_code += 1
                line_num += 1
            print(str(file_num) + " :", end="")
            print(path + "/" + file)
            file_num += 1
        except:
            pass
        f.close()
print("""
你一共写的代码行数为：{}
其中空行为：{}
注释行数为：{}
代码行数为：{}
""".format(line_num, line_space, line_annotate, line_code))
