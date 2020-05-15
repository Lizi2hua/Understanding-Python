# 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
#
# 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
#
# 你可以假设除了整数 0 之外，这个整数不会以零开头
# 输入: [1,2,3]
# 输出: [1,2,4]
# 解释: 输入数组表示数字 123。
list = [1, 2, 3]
# 特殊情况：最后一位会进位[9,9,9,9,9]=99999=100000
# 通常情况,即没有进位的情况
if list[-1] != 9:
    list[-1] += 1
    print(list)
# 有进位情况
# [......x,9]
# [....x,9,9]
# [....x,9,9,9]
# 全为9为有进位情况的子集,但需要改变数组维度
# 从特殊情况出发[9,9,9,9,9,9,9,9,9,9,9,9]这种,用指针,
i = 1
j = 3
for j in range(3):
    print(list[-i])
    i += 1
# 从后往前遍历
# [x,x,x,x,x,9,9,9,9,9]
testlist = [9, 9, 9, 9, 9]
len = len(testlist)
k = 1
for j in range(len - 1):
    while testlist[-k] == 9:
        testlist[-k] = 0
        k += 1
        if k == len:
            break
    # 数组维度+1
    if testlist[0] == 9:
        testlist[0] = 0
        testlist.insert(0, 1)
    else:
        testlist[-k] += 1
    break
print(testlist)


# 封装
def plusOne(nums):
    l = len(nums)
    if nums[-1] != 9:
        nums[-1] += 1
    else:
        k = 1
        for j in range(l - 1):
            while nums[-k] == 9:
                nums[-k] = 0
                k += 1
                if k == l:
                    break
            # 下面代码可以放在外面?
            if nums[0] == 9:
                nums[0] = 0
                nums.insert(0, 1)
            else:
                nums[-k] += 1
            break
    return nums


testlist2 = [8, 9, 9, 9, 9, 9]
testlist3 = [1, 2, 2, 3, 1, 3, 9]
testlist4 = [9, 9, 9, 9, 9]
print(plusOne(testlist2))
