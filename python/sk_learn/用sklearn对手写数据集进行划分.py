import sklearn
from sklearn.model_selection import train_test_split
import os
import json

PATH=r'C:\Users\Administrator\Desktop\dataset\street\mchar_train'
LABEL=r'C:\Users\Administrator\Desktop\dataset\street\mchar_train.json'
dir=os.listdir(PATH)
print(dir)
with open(LABEL,'r',encoding='utf-8') as fp:
    label=json.load(fp)
    label_key=list(label.keys())
#
print(dir)
print(label_key)
x_train,x_test,y_train,y_test=train_test_split(dir,label_key,random_state=33)
print(len(x_train))
print(len(x_test))

