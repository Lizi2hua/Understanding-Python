from sklearn import svm,datasets
from sklearn.metrics import f1_score,confusion_matrix,classification_report
from sklearn import  model_selection
# 数据处理
Data=datasets.load_iris()
data=Data['data']
target=Data['target']

train_data,test_data,train_target,test_target=model_selection.train_test_split(data,target,test_size=0.25)
# 模型训练
clf=svm.SVC()
clf.fit(train_data,train_target)
# 模型评估
pred=clf.predict(test_data)
print(test_target)
print(pred)
print(f1_score(test_target,pred,average='micro'))
print(confusion_matrix(test_target,pred))
print(classification_report(test_target,pred))
