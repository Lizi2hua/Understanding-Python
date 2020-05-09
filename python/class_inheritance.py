#类的继承
#1单继承
#定义一个父类
class Person(object):
    #定义构造函数
    def __init__(self,name,age):
        self.name=name
        self.age=age
        print("he/she's name is:",name,"and age is:",age)
    def talk(self,word):
        print("people is talking:",word)
#定义一个继承Person类的子类
class Chinese(Person):
    #继承后重构父类的构造函数
    def __init__(self,name,age,language): #参数必须有父类的参数，否则报错（因为下行继承的父类函数必须有这些参数）
        Person.__init__(self,name,age)#继承父类的构造函数，也可以用super
        self.lang=language
    #with_chiness=True为什么传不进去？
    # 这么传
    # with_chinese=True
    # def talk(self,word,with_chinese):
    #     self.with_chinese=with_chinese
    #     if(with_chinese):
    #         print("is talking:",word,"with chinese")
    #     else:
    #         print("is talking:",word,"without chinese" )
    def talk(self,word,with_chinese=True):
        if(with_chinese):
            print("is talking:",word,"with chinese")
        else:
            print("is talking:",word,"without chinese" )
class American(Person):
    pass
#pass 是占位符，为了保存结构完整性。
# wang=Chinese(name="wang",age=24,language="chinese")
# wang.talk(word="wdnmd")

#2.多重继承
class people(object):
    name=''
    age=0
    # 以双下划线表示私有属性__weight
    __weight = 30
    def __init__(self,name,age,weight):
        self.name=name
        self.age=age
        self.__weight=weight
    def speak(self):
        print("%s is speaking : i am %d yrs old"%(self.name,self.age))

class student(people):
    grade=''
    #重写构造函数
    def __init__(self,name,age,weight,grade):
        people.__init__(self,name,age,weight)
        self.grade=grade
    def speak(self):
        print("my name is %s,i am %d yrs old,and i am in %s grade"%(self.name,self.age,self.grade))

class speaker():
    topic=''
    name=''
    def __init__(self,name,topic):
        self.name=name
        self.topic=topic
    def speak(self):
        print("i am %s,my topic is %s"%(self.name,self.topic))



