# 集合的特点：
#     1.元素不可变,不能改
#     2.天生去重
# 因为每存一个值到set里时， 都要先经过hash，然后通过得
# 出的这个hash值算出应该存在set里的哪个位置，存的时候会
# 先检查那个位置上有没有值 ，有的话就对比是否相等，如果相
# 等，则不再存储此值。 如果不相等(即为空)，则把新值 存在这。

#     3.无序，{1，2，3}和{3，2，1}算同一个集合

a={1,1,2,2,3,6,78,5,88,415,22,231468}
print(a)
# {1, 2, 3, 5, 6, 231468, 78, 22, 88, 415}
# ********************************
# 帮列表去重
list=[1,2,3,4,5,4,3,2,'alex','susan','alex']
print("list:",list)
print("去重后:",set(list))
# list: [1, 2, 3, 4, 5, 4, 3, 2, 'alex', 'susan', 'alex']
# 去重后: {1, 2, 3, 4, 5, 'susan', 'alex'}
# ********************************
# 增
a.add('allen')
#删
a.discard(2)
#查
print('alex' in a)
#改
# 不能改