#列表的操作
#1.追加到尾部
names=[]
for i in range(10):
    names.append(i)
    print(names)
print("------------------------------")
#迭代器
print("use iterator to generate list")
list1=[]
print("1.use list to generate list")
list2=[2,3,4,5,6,7,8,9]
for i in list2:
    list1.append(i)
print(list1)
print("--------------------------")
list3=[]
print("use string to generate list")
string="asss"
for i in string:
    list3.append(i)
print(list3)
print("--------------------------")
list4=[]
print("use dictionary to generate list")
diction={"aaa":222,"cde":233,"dsd":2323}
for i in diction:
    list4.append(i)
print(list4)
