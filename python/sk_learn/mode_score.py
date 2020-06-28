from sklearn.metrics import explained_variance_score,r2_score,mean_squared_error,mean_absolute_error
# 可解释方差
y_true=[3,-0.5,2,7]
y_pred=[-10,0.0,0,10]
print("explained error:")
print(explained_variance_score(y_true,y_pred))
print(explained_variance_score(y_true,y_pred,multioutput='raw_values'))
print("r2 score:")
# R2评分
print(r2_score(y_true,y_pred))
print(r2_score(y_true,y_pred,multioutput='raw_values'))
# 均方误差，平均绝对误差
print(mean_squared_error(y_true,y_pred))
print(mean_absolute_error(y_true,y_pred))