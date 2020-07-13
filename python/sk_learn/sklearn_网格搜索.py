from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np

iris_data=load_iris()
X_trainval,X_test,y_trainval,y_test = train_test_split(iris_data.data,iris_data.target,random_state=0)
X_train,X_test,y_train,y_test = train_test_split(X_trainval,y_trainval,random_state=1)
#得到[train,val,test]数据集

#实现Grid Search with CV
# best_score=0
# for gamma in [0.001,0.01,1,10,100]:
#     for C in [0.001,0.01,1,10,100]:
#         svc=SVC(gamma=gamma,C=C)#gamma是核系数，C是正则化参数
#         #5折交叉验证
#         scores=cross_val_score(svc,X_trainval,y_trainval,cv=5)
#         print(scores)#-->[0.34782609 0.34782609 0.36363636 0.36363636 0.40909091]
#         score=scores.mean()
#         #找到表现最好的参数
#         if score > best_score:
#             best_score=score
#             best_parameters={'gamma':gamma,'C':C}
# print(best_parameters)
# #使用最佳参数，构建新的模型
# svc=SVC(**best_parameters)

# 使用GridSearchCV
param_grid={'gamma':[i for i in np.arange(0.1,10,0.01)],'C':[i for i in np.arange(0.1,100,1)]}
grid_search=GridSearchCV(SVC(),param_grid,cv=6,n_jobs=-1)
grid_reult=grid_search.fit(X_trainval,y_trainval)#运行网格搜索
print(grid_reult.best_score_)
print(grid_reult.best_params_)

