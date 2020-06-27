import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import  make_regression
# coef:斜率,生成回归数据
x,y,coef=make_regression(n_samples=100,n_features=1,noise=10,coef=True)
print(coef)
plt.scatter(x,y)
plt.show()