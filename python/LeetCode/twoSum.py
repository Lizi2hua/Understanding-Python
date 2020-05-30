"""给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。"""
"""你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。"""
nums = [2, 3, 4, 5, 6, 12, 2, 5, 5, 6]

target = 100

num1 = []
num2 = []

for i in range(len(nums)):
    num1 = nums[i]

    for j in range(len(nums)):
        num2 = nums[j]

        if num1 + num2 == target:
            break
    if i == len(nums):
            print("not found")

print("index:", "[", i, ',', j, ']')
"""未完成！"""