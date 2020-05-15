# 断言用于判断一个表达式，在表达式条件为false的时候触发异常
# 可以在不满足程序运行情况下直接返回错误，而不必等待程序运行后崩溃的情况
# 语法 assert expression
# 判断系统是否为linux
import sys

# print(sys.platform)
# assert ('win32' in sys.platform), '平台错误'
c = [1,2,3,4,5]
assert ('3' in c), "c不能有3"
