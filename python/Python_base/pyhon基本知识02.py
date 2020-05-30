"""在 Python 中，你也可以定义包含若干参数的函数。这里有三种可用的形式，也可以混合使用。"""
def ask_ok(prompt,retries=4,complaint='yes or no,please！'):
    while True:
        ok=input(prompt)
        if ok in ('y','ye','yes'):
            return True
        if ok in ('n','no','nope'):
            return False
        retries=retries-1
        if retries<0:
            raise OSError('uncooperative user')
        print(complaint)
# ask_ok('do you really want to quit:')

# def f(a,L=None):
def f(a,L=[]):
    # if L is None:
    #     L=[]
    L.append(a)
    return L
# print(f(1))
# print(f(2))
# print(f(3))
# 和变量的作用域有关?参数列表的作用域在函数外面？还是说和垃圾回收机制有关？
def test(a,L=[]):
    L.append(a)
    return L
# print(f(1))
# print("==============")
# print(test(4))


