# 机器学习

## 1.L1正则化（Regularization）与L2正则化

​		所谓的正则化就是在原来的*loss function*基础上加一项模型复杂度的惩罚项。模型的复杂度可以用*VC维*来衡量，VC维度越大，则学习过程越复杂。

优化目标：
$$
min_w||Xw-y||_2^2
$$
L1正则化：
$$
min_w||Xw-y||_2^2+\alpha||w||_1
$$
L2正则化：
$$
min_w||Xw-y||_2^2+\alpha||w||_2^2
$$


​		VC 维是衡量函数类的复杂度的一种方式，通过**评估函数类中函数的弯曲程度**实现。在平面中，存在两类点，一类点类别为0，另一类为1。如果VC维等于3，即平面任意3个点总能被一条直线分开，4个点却不行。**一般地**，***p***维线性指示函数VC维为***p+1***。

<img src="C:\Users\李梓桦\Desktop\培训V20200507\CODE\python\Note\src\line_cv.png" alt="line_cv" style="zoom:80%;" />

​		下图是无穷VC维，对于实轴上的任意多个点，总存在一个函数能将他们分开。

<img src="C:\Users\李梓桦\Desktop\培训V20200507\CODE\python\Note\src\vc_n.jpg" alt="vc_n" style="zoom: 80%;" />

### 1.1 从结构风险最小化角度

​		结构风险最小化（Structural Risk Minimizatiom）的基本思想是在保证**分类精度(经验风险)**时，降低模型的VC维才能取得较小的实际风险，即对未来的样本有较好的推广性。以w1和w2组成的解的空间为例：

<img src="C:\Users\李梓桦\Desktop\培训V20200507\CODE\python\Note\src\L1_L2regualarzition.jpg" alt="L1_L2regualarzition" style="zoom:80%;" />



对于L1正则化，如右图所示：

- 优化需要同时最小化两项。如果不加L1正则化得话，优化结果为圆圈内部紫色的部分。对于w1和w2来说，值域所围成的形状为菱形（|w1|+|w2|=F）。对于红色曲线，每一点都可以做一个菱形(曲线上每个w1,w2的取值都可以确定一个菱形)。由图可见，当w2=0时候，两参数确定的菱形最小。**这也是L1更容易得到稀疏解得原因。**

<img src="C:\Users\李梓桦\Desktop\培训V20200507\CODE\python\Note\src\l1.jpg" alt="l1" style="zoom:50%;" />

- L2正则化同理，不过L2正则化不容易交于坐标轴上，但值仍然会靠近坐标轴。



### 1.2 从概率论的角度

​		**L1服从拉普拉斯分布，L2服从高斯分布。**（具体原因看链接）

​		先验就是优化的起跑线, 有先验的好处就是可以在较小的数据集中有良好的泛化性能，当然这是在先验分布是接近真实分布的情况下得到的了，从信息论的角度看，向系统加入了正确先验这个信息，肯定会提高系统的性能。

拉普拉斯分布：
$$
f(x)=\frac{1}{2\lambda}e^{-\frac{|x-\mu|}{\lambda}},\lambda，\mu为常数，且\lambda为常数
$$


​		如果ω服从标准拉普拉斯分布，那么ω取0的概率非常大。

​                                     <img src="C:\Users\李梓桦\Desktop\培训V20200507\CODE\python\Note\src\laplace.jpg" alt="laplace" style="zoom:80%;" />	

​		如果ω服从标准高斯分布，那么ω取0附近值的概率非常大。

<img src="C:\Users\李梓桦\Desktop\培训V20200507\CODE\python\Note\src\gauss_dis.png" alt="gauss_dis" style="zoom:80%;" />

### 1.3 总结

​		正则化能够降低模型的结构风险，通俗来说就是将模型变得相对简单，这符合*奥卡姆剃刀*理论。

​		L1正则化就是在loss function后边所加正则项为L1范数，加上L1范数容易得到稀疏解（0比较多）。L2正则化就是loss function后边所加正则项为L2范数的平方，加上L2正则相比于L1正则来说，得到的解比较平滑（不是稀疏），但是同样能够保证解中接近于0（但不是等于0，所以相对平滑）的维度比较多，降低模型的复杂度。

​		

## 3.回归（Regression）

​		回归，事物总是朝着某种**平均**发展，也可以说是朝着事物得某种本来得面目发展。

### 3.1 什么是回归

​		回归是一种建立预测模型的方法，这种方法常用于预测分析，时间序列模型以及变量之间的因果关系。在回归任务中，计算机程序需要对给定输入预测输出数值。*除了返回结果的形式不一样外，这类问题和分类问题是很像的。***若我们预测的是离散值，此类任务为`分类`，若我们预测的是连续值，此类任务为`回归`**

### 3.2 回归模型[^1]

​		常用的回归模型有`线性回归（Linear Regression）`,`岭回归（Ridge Regression）`,`LASSO回归`，`弹性网络（ElasticNet）`，`核岭回归（Kernel Ridge Regression,KRR）`

#### 3.2.1  广义线性模型

​		目标值$y$是输入变量$x$的线性组合。如果有有$\hat{y}$是预测值，则满足：
$$
\hat{y}(w,x)=w_0+w_1x_1+...+w_px_p
$$
可见，在线性方程中，$w=(w_1,...,w_p)$为系数（coef）向量，$w_0)$为截距。

​		线性回归模型通过`最小二乘法（Least Square）`来拟合$w=(w_1,...,w_p)$

的线性模型，使得数据集实际数据（labeled）和预测数据（估计值）之间的残差[^2]平方和最小，即
$$
min_w||Xw-y||_2^2
$$
普通最小二乘法的复杂度：设$$n_{samples} \geq n_{features}$$,则复杂度为$$O(n_{samples} n_{fearures}^2)$$

```python
from sklearn import  linear_model
x=[[0,0],[1,1],[2,2]]
y=[0,1,2]
model=linear_model.LinearRegression()
model.fit(x,y)
print(model.coef_)
print(model.intercept_)
```

### 3.2.1 岭回归





## x. scikit-learn



![sklearn_model_select](C:\Users\李梓桦\Desktop\培训V20200507\CODE\python\Note\src\sklearn_model_select.jpg)

### x.1 使用sklearn构建机器学习模型的基本步骤

```python
from sklearn import  neighbors,datasets,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  cross_val_score

# 加载数据
iris=datasets.load_iris()

# 划分训练集和测试集
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=33)
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3
#  radom_state是随机种子

# 数据预处理
scaler=preprocessing.StandardScaler().fit(x_train)
# 根据x_train得到scaler--标准化的方法

#标准化，即标准正态分布化
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# 创建模型
knn=neighbors.KNeighborsClassifier(n_neighbors=5)

# 模型拟合
knn.fit(x_train,y_train)

# 交叉验证
scores=cross_val_score(knn,x_train,y_train,cv=5,scoring="accuracy")
print(scores)
print(scores.mean())

# 预测（使用）
y_pred=knn.predict(x_test)
# 评估
print(accuracy_score(y_test,y_pred))
```



------

[^1]: http://www.scikitlearn.com.cn/0.21.3/2/

[^2]: 残差（residual）是指实际观测值与估计值（拟合值）之间的差。残差蕴含了有关模型基本假设的重要信息。如果回归模型正确的话， 我们可以将残差看作误差的观测值。利用残差所提供的信息，来考察模型假设的合理性及数据的可靠性称为残差分析。





------

[sk-learn中文文档]：http://www.scikitlearn.com.cn/

[正则化]：https://zhuanlan.zhihu.com/p/35356992?utm_medium=social&utm_source=wechat_session

[VC维]https://www.zhihu.com/question/23418822/answer/299969908

