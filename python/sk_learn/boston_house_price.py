import numpy as np
from sklearn import  datasets
from sklearn import  model_selection
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# load data
src=np.loadtxt('housing.data')
price=src[:,13]
data=src[:,0:12]

# split data
train_data,test_data,train_price,test_price=model_selection.train_test_split(data,price,test_size=0.25)
# print(train_data)
# print(test_data)
# print(train_price)
# print(test_price)
# fit data
tuned_paras=[{'kernel':['rbf','laplacian','polynomial'],'gamma':[1e-3,1e-4],
              'alpha':[1e0,0.1,1e-2,1e-3]
              } ]
scores=['explained_variance','r2']

for score in scores:
    print("fit data with {} score".format(score))

    model=GridSearchCV(KernelRidge(),tuned_paras,cv=5,scoring=str(score))
    model.fit(train_data,train_price)

    print("i find the best params in tune_params")
    print(model.best_params_)
print('test socre')
pred_price=model.predict(test_data)
print(r2_score(test_price,pred_price))