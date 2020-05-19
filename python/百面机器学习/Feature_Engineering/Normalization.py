import numpy01

# !!!只针对一维list情况
# 数据类型归一化
# 1.线性函数归一化（Min_Max Scaling)
def min_max_scaling(list):
    # 寻早 min max
    min_list = min(list)
    max_list = max(list)
    new_list = []
    for i in range(len(list)):
        tmp = (list[i] - min_list) / (max_list - min_list)
        new_list.append(tmp)
    return new_list


# list=[1,2,3,4]
# new_list=min_max_scaling(list)
# print(new_list)
# pass
def z_score_norm(list):
    mean=numpy01.mean(list)
    std=numpy01.std(list)
    new_list=[]
    for i in range(len(list)):
        tmp=(list[i]-mean)/std
        new_list.append(tmp)
    return new_list
# list=[1,2,3,4]
# new_list=z_score_norm(list)
# print(new_list)
# print("mean:",numpy.mean(new_list))
# print("std:",numpy.std(new_list))
# pass