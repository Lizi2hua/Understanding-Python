#元组是一种只读列表，不能修改
data=(2,3,1,5,1,2)
#如果元组中还包含其他可变元素，这些可变元素可变
data2=(22,3,5,["alex","jack"],3)
print(data2)
print("                      |")
#修改元组中的列表的值
data2[3][1]="rose"
print(data2)
# 原因：元组只存放每个元素的内存地址，不变指内存地址不变，
# 但是上例中，元组存放的是列表["alex","jack"]这一列表的内存地址
# 列表内的元素存在列表内存的内存地址中，所以可变。