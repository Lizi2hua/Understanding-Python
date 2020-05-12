# 给定一个整数类型的数组 nums，请编写一个能够返回数组“中心索引”的方法。
#
# 我们是这样定义数组中心索引的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。
#
# 如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。
# 采用一个指针，
nums = [1, 7, 3, 6, 5, 6]
code=-1

# for i in range(len(list)):
#     print(list[i])
#     # i为0的情况
#     # if i == 0:
#     #     pass
#     # else:
#     # list切片是左闭右开
#     sum_left = sum(list[0:i])
#     sum_right = sum(list[i + 1:])
#     if sum_left==sum_right:
#         code=i
# print("i find it in:%d" % code)

#
# print(list[4:])
# sum_list = sum(list)
# print(sum_list)
sums=sum(nums)
left_sum=0
for i in range(len(nums)):
    if nums[i]==sums-left_sum*2:
        print("i find it in %d"%i)
    left_sum+=nums[i]




