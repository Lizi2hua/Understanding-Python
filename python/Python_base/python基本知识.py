"""python的/y永远返回浮点"""
a = 25
b = 5
print(type(a / b))
"""floor除法, //得到整数结果"""
a1 = 7
a2 = 3
print(a1 // a2)
# 2
# 7//3.5=2.0
"""转义"""
print("\"Yes,\"he said")

print("C:\some\name")
# \n是换行符，使用r来使用原始字符串
print(r"C:\some\name")

"""使用 \ 作为连续符"""

print("liu\
mengyuan")
# liumengyuan
"""或者这样"""
print('liu'
      ' mengyuan')
# liu mengyuan

word = "python"
# 当索引的右边值大于实际长度时，会被字符串的实际长度代替
print(word[3:44])
# 左边过大返回空
print(word[22:])
# list也适用
lis = [0, 1, 2, 3, 4, 5]
print(lis[2:22])
"""序列类型,可以索引的类型"""
set = (0, 2, 3, 4, 4, 2, 3, 4)
print(set[1:5])

squares = [1, 3, 5, 6, 7, 10]
b = [2, 2, 2]
print(type(squares + b))

"""变量由右边赋值,右边先计算"""
a, b = 0, 1
while b < 10:
    print(b)
    a, b = b, a + b
