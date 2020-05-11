# 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
#
# 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
#
# 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
val = 6
list = [3, 5, 6, 1, 6, 8, 1, 3, 1, 1, 6, 8, 1, 2, 4, 6, 1, 3, 2, 3]
# 1找到等于val的元素
len_l = len(list)
for i in range(len_l):
    if list[i] == val:
        print("i find it in list[%d]" % i)
# 2.原地算法移除相等的元素
list2 = [3, 5, 6, 1, 6, 8, 1, 3, 1, 1, 6, 8, 1, 2, 4, 6, 1, 3, 2, 6]
i = -1
j = 0
for n in range(len_l):
    if list2[j] != val:
        i += 1
        list2[i] = list2[j]
        print(list2)
    j += 1
    # 新长度就是i的值

    print(i)


# 整理成函数
def removeElement(nums, val):
    len_n = len(nums)
    i = -1
    j = 0
    for n in range(len_n):
        if nums[j] != val:
            i += 1
            nums[i] = nums[j]
        j += 1
    print(nums[0:i])
removeElement(list2,val)