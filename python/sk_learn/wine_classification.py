import numpy as np
from sklearn import  datasets
from sklearn import  model_selection
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV

# load data
src=np.loadtxt('wine.data',delimiter=',')
# print(src)
data=src[:,1:]
target=src[:,:1]
target=np.array(target).ravel()
# print(data)
# print(target)

# split data
train_data,test_data,train_target,test_target=model_selection.train_test_split(data,target,test_size=0.3)
# print(train_data)
# print(test_data)
# print(train_target)
# print(test_target)
# fit data
tuned_paras=[{'kernel':['rbf','poly'],'gamma':[1e-3,1e-4],
              'C':[1e3,1e2,1e1,1,10,100]
              }]

model=GridSearchCV(SVC(),tuned_paras,cv=5)
model.fit(data,target)
print("i find the best params in tune_params")
print(model.best_params_)

pred_target=model.predict(test_data)
print("reacll score")
print(recall_score(test_target,pred_target,average='macro'))
print(recall_score(test_target,pred_target,average='micro'))
print(recall_score(test_target,pred_target,average='weighted'))
