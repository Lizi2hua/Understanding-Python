# https://docs.pythontab.com/python/python3.5/controlflow.html"""
"""在迭代时修改迭代序列不安全"""
w = ['cat', 'windows', 'dog']
for i in w:
    print(i)
    if len(i) < 4:
        # w.append('mew')
        # 无限循环
        w.append('mewo')
# print(w)

"""可以用它的副本"""
for i in w[:]:
    print(i)
    if len(i) < 4:
        w.append('mew')
print(w)


"""不同方面 range() 函数返回的对象表现为它是一个列表，但事实上它并不是。当你迭代它时，它是一个能够像期望的序列返回连续项的对象；但为了节省空间，它并不真正构造列表。
我们称此类对象是 可迭代的，即适合作为那些期望从某些东西中获得连续项直到结束的函数或结构的一个目标（参数）。"""


