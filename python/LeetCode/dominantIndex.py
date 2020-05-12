# 在一个给定的数组nums中，总是存在一个最大元素 。
#
# 查找数组中的最大元素是否至少是数组中每个其他数字的两倍。
#
# 如果是，则返回最大元素的索引，否则返回-1。
nums = nums = [3, 6, 1, 0]
max = nums[0]
# 1.找到最大元素
for i in range(len(nums)):
    if nums[i] >= max:
        max = nums[i]
        max_index=i
        print("the new max value is %d,at nums[%d]" % (max, i))
    else:
        pass
print(max)
# 2。生成新数组，×2
# a.有多个最大值
for j in range(len(nums)):
    if nums[j]!=max:
        d_num=2*nums[j]
        if max<d_num:
            print("不满足条件")
        else:pass
    else:pass
   #每个元素的两倍




# 封装成函数
#def dominantIndex(nums)
