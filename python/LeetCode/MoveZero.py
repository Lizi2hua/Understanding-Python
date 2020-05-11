#   移动零
# 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
# 示例:
#
# 输入: [0,1,0,3,12]
# 输出: [1,3,12,0,0]

# 如何找到0元素
# 1.
list = [0, 1, 2, 4, 32, 7, 0, 72, 1, 0, 4, 8]
for i in range(len(list)):
    if list[i] == 0:
        print("i find zero! in list[%d]" % i)
# 2.list.index(),找出某个值的第一个匹配项的索引位置


# 如何移动list中的元素
list2 = [0, 1, 2, 4, 32, 7, 0, 72, 1, 0, 4, 8]
zero_index = 6
zero_tmp = list2.pop(zero_index)
print(list2)
# list.insert(index,obj)
list2.append(zero_tmp)
print(list2)

# 将所有的0找出来，删除所有的0，将所有的0增加到末尾
list3 = [0, 1, 2, 4, 32, 7, 0, 72, 1, 0, 4, 8]
list_zero = []
zero = 0;
len_list = len(list3)
# find 0
for i in range(len_list):
    if list3[i] == 0:
        zero += 1
for i in range(zero):
    list3.remove(0)
    list3.append(0)
print(list3)


def moveZeroes(list):
    zero = 0
    len_list = len(list)
    # find 0
    for i in range(len_list):
        if list[i] == 0:
            zero += 1
    for j in range(zero):
        list.remove(0)
        list.append(0)
    print(list)

list4=[0,5,0,4,78,0,0,4,0,9,0,6,7]
print(list4)
list_changed=moveZeroes(list4)
print(list_changed)
# TypeError
