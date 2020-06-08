# https://www.runoob.com/python/python-exceptions.html
#1.语法错误SyntaxError
#2.异常处理 try/except
while True:
    try:
        x=int(input("enter a number:"))
        print("you number is:{}".format(x))
        break
    except ValueError:
        print("enter an int value!try again!")
#3.使用raise抛出异常
x=10
if x>5:
    raise Exception('x不能大于5')