from sklearn import preprocessing
import  numpy as np
from sklearn.impute import  SimpleImputer

X_train=np.array([[1.,-1,2],
                  [2.,0,0],
                  [0,1,-1.]])
y_train=np.array([[1.,-2,2],
                  [2.,0,1],
                  [0,2,-2.]])

scaler=preprocessing.StandardScaler().fit(X_train)
# print(scaler)
y_train=scaler.transform(y_train)
# print(y_train)
X_train=np.array([[1.,-1,2],
                  [2.,0,0],
                  [0,1,-1.]])
x_sacled=preprocessing.scale(X_train)
# print(x_sacled)
#
# X_norm1=preprocessing.normalize(X_train,norm='l2')
# print(X_norm1)
# X_norm2=preprocessing.normalize(X_train,norm='l1')
# print(X_norm2)
# X_norm3=preprocessing.normalize(X_train,norm='max')
# print(X_norm3)
normalizer=preprocessing.Normalizer().fit(X_train)
# print(normalizer.transform(y_train))

# 缺失值处理
imp1=SimpleImputer(missing_values=np.nan,strategy='mean')
y_imp1=imp1.fit_transform([[np.nan,2],[6,np.nan],[7,6]])
print(y_imp1)
imp2=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
y_imp2=imp2.fit_transform([[np.nan,2],[6,np.nan],[7,6]])
print(y_imp2)