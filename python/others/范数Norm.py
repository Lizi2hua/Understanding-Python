import numpy as np

# def Nor(array,n=2)
#     """n值为n范数,默认为欧式距离"""
array = np.array([-1, 2, -3, 3, 5, 6, 2, -2, 5, 6])
# n=2
# for i in range(len(array)):
#     array[i]=abs(array[i])
#     array[i]=array[i]**n
#     sum_arr=sum(array)
#     sqrt_arr=sum_arr**(1/n)
# print(sqrt_arr)

"""0范数，无穷范数不知怎么实现，故只实现2范数，即欧式距离"""
def l2Norm(array):
    n = 2
    for i in range(len(array)):
        array[i] = abs(array[i])
        array[i] = array[i] ** n
        sum_arr = sum(array)
        sqrt_arr = sum_arr ** (1 / n)
    return sqrt_arr

print(l2Norm(array))

