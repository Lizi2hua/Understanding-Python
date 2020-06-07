name="Eva"
age=16
sex=None
#*******************#
if name=="Eva":
    sex="female"
else:
    sex="male"
print(sex)
#*******************#
#用三元运算写
sex2="female"if name=="Eva"else"Male"
print(sex2)
#*******************#
#嵌套的if-else
age2=60
if age2<=18:
    status="child"
else:
    if age2<=50:
        status="young"
    else:
        status="old"
print(status)
#*******************#
status2="child" if age2<=18 else "young" if age2<=50 else "old"
print(status2)