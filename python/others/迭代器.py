# 迭代器是一个可以记住遍历的位置的对象
# 迭代器从集合的第一个元素开始访问，只能向前不能往后，知道所有的原素被访问完。
list=[1,2,3,4]
it=iter(list)
for i in range(len(list)):
    print(next(it))