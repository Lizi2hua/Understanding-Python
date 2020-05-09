# #变量创建过程：
#     先申请一块存储空间来存储数据
#     将变量名指向这一块空间
name="old-data"
print("name addr",id(name))
name2=name
print("name2 addr",id(name2))
#************#
data3="old"
data4=data3
print(data3,id(data3)," ",data4,id(data4))
data3="new"
print(data3,id(data3)," ",data4,id(data4))
# output:
#     name addr 2215906459952
#     name2 addr 2215906459952
#     old 2215905089520   old 2215905089520
#     new 2215905089456   old 2215905089520