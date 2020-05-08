
#https://www.yiibai.com/python/python_classes_objects.html
#self用来代替将会实例化的类的类本身（self.name-->student.name)
class Employee:
    "commom base class for all employee"
    empCount=0

    def __init__(self,name,salary):
        self.name=name
        self.salary=salary
        Employee.empCount+=1
# __init__()是构造函数，可选，如果不提供不默认给出__init__方法
#    与之相对,__del__为析构函数
    def displayCount(self):
        print("total employee",Employee.empCount)
    def displayEmp(self):
        print("name:",self.name," ,Salary",self.salary)

n1=Employee("susi",2000)
Employee.displayCount(n1)
Employee.displayEmp(n1)
#https://code.ziqiangxuetang.com/python3/python3-class.htmlhttps://code.ziqiangxuetang.com/python3/python3-class.html
#继承 class DerviedClassName(BaseClassNaem):xxxx
class Manager(Employee)



#多重继承 class DerviedClassName(Base1,Base2,...)
