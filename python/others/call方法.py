# __init__；__new__;__call__
class A(object):
    def __init__(self):
        super(A, self).__init__()
        print("doing __init__")
        # __init__无返回值

    # 是一个静态方法
    def __new__(cls):
        print("doing __new__")
        return super(A, cls).__new__(cls)

    def __call__(self):
        print("doing __call__")


A()
# 执行顺序
# doing __new__
# doing __init__
# __new__的返回值为类的实例对象，该实例对象会传递给__init__方法中定义的self参数，让对象初始化
# 如果没有返回值，那么就只执行到 __new__，不会执行__init__
