"""解包赋值是一个方便代码编写的语法特性，在一个赋值语句中可将右边表达式的值自动解开赋值给左边"""
"""左边变量个数得与右边对象元素个数相等"""
t = [1, 2]
a, b = t
# print(a,b)
# 1 2
t1 = "我你"
a1, b1 = t1
# print(a1,b1)
# 我 你
t2 = {5: "w22", '23333': 2}
a2, b2 = t2
# 赋键的值
# print(a2,b2)

"""参数列表的分拆"""

# 在变量前面加一个 * 操作符来将【列表】自动分拆成参数列表
args = [1, 10, 2]


# print(list(range(*args)))
# [1, 3, 5, 7, 9]
# 使用 ** 操作符分拆【字典】 keyword:value
def parrot(voltage, state='a stiff', action='voom'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.", end=' ')
    print("E's", state, "!")
d={'voltage':'full','state':"bleeddin'demised ","action":"Voom"}
parrot(**d)