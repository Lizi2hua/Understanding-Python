staff_list=[
    ["alex",22,'ceo'],
    ["cider",23,'hr']
]
for i in staff_list:
    print(i)
# 用列表的方式存储这种数据，速度很慢。
# dic是python中唯一的映射类型
# 定义{key1:value1,key2:value2}，key值必须唯一且不变
dic1={"key1":"value1","key2":"value2"}
print(dic1["key1"])
# 一个key可以有任意多个value，value可修改、可以不唯一
dict2={"name":["alex","lisa","yuki"],"age":[21,22,23]}
print(dict2["name"])
# key可以对应任意多个value的本质应该是key对应的value是以存储地址存在。
print('----------------------------')
dict3={"alex":[23,'ceo',32600],'susan':[22,'cfo',22335]}
#新增
dict3['yui']=[20,'hr',25000]
print(dict3)
print('----------------------------')
import random
# #字典的拼接
# ****************************************#
dict4={}.fromkeys([12,3,4],2)
dict5={}.fromkeys(random.sample('zzsdadadadqwrwrfsfhgrbejrg',5),random.sample('zzsdadadadqwrwrfsfhgrbejrg',1))
dict4.update(dict5)
print(dict4)
print('----------------------------')
#删除
#删除指定kye
dict4.pop(12)
print(dict4)
#修改
dict4[4]='zzz'
print('----------------------------')
#for循环
for i in dict4:
    print(i,dict4[i])
print('-----------exercise----------------')
#****************************************#
#exercise
#{‘k0’: 0, ‘k1’: 1, ‘k2’: 2, ‘k3’: 3, ‘k4’: 4, ‘k5’: 5, ‘k6’: 6, ‘k7’: 7, ‘k8’: 8, ‘k9’: 9}
# 请把这个dict中key大于5的值value打印出来。
dict_tmp={'k0':0,'k1':1,'k2':2,'k3':3,'k4':4,'k5':5,'k6':6,
          'k7':7,'k8':8,'k9':9}
for i in dict_tmp:
    value_tmp=dict_tmp[i]
    if value_tmp>5:
        print(value_tmp)
    else:pass
