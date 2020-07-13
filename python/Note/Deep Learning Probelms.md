# Deep Learning Probelms

## 1. SGD为什么是个很好的方法[^1]

​		普通GD算法：
$$
w_{t+1}=w_t-\eta\bigtriangledown l(x_t)=w_t-\eta\bigtriangledown(x_t*w_t-\hat{y})
$$
​		GD算法每次都是通过整个数据集来计算损失函数的梯度这意味着如果数据集太大，那么梯度计算将会变得很慢，最后还只能走一小步，一般GD需要走很多步才能走完。

​		***使用整个训练集称为批量（batch），术语批量梯度下降（BGD）指使用全部数据集，当批量单独出现时指使用一组样本。* **

​		其次如果进入鞍点，或者比较差得全局最优点，因为这些点的导数为0，GD算法就跑不出来。

![paddle_point](C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\paddle_point.jpg)

​		SGD算法：
$$
w_{t+1}=w_t-\eta\bigtriangledown g_t(w,b)
$$
​		$g_t$称为随机梯度，满足$E[g_t]=\bigtriangledown l(x_t)$,即虽然梯度包含一定的随机性，但是**从期望看，它是等于正确的导数（使用GD时的导数）。更大的批量拥有更精确的梯度估计。**

​		***每次只使用单个样本的优化算法称为随机或在线算法，从连续产生的样本数据流中抽取样本***

​	大多数用于深度学习的算法介于以上两者之间，使用一个以上而又不是全部的训练样本。**传统上，这些会被称为小批量 （minibatch）或小批量随机 （minibatch stochastic）方法，现在通常将它们简单地称为随机 （stochastic）方法。**

​	随机梯度下降法，不像BGD每一次参数更新，需要计算整个数据样本集的梯度，而是**每次参数更新时，仅仅选取一个样本计算其梯度。**

​		用一张图来表示，其实SGD就像是喝醉了酒的GD，它依稀认得路，最后也能自己走回家，但是走得歪歪扭扭．（红色的是GD的路线，偏粉红的是SGD的路线）．

![SGD_GD](C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\SGD_GD.png)

 		  SGD（指mini batch SGD）由于在小批量的学习过程中加入了噪声，这些噪声会产生正则化的效果。

​		实践中，人们发现，除了算得快，SGD有非常多的优良性质．它能够自动逃离鞍点，自动逃离比较差的局部最优点，而且，最后找到的答案还具有很强的一般（generalization），即能够在自己之前没有见过但是服从同样分布的数据集上表现非常好！

​		

## 2. 激活函数

​		激活函数**给神经网络引入了非线性的能力，满足连续可导（条件性满足）。**

### 2.1 Introduction

​		如果没有激活函数：
$$
f_{11}(x_1,x1_2)=0.3*x_1+0.4*x_2+3\\
f_{12}(x_1,x_2)=0.6*x_1+0.6*x_2+2\\
f_{21}(f_11,f_12)=0.5*f_11+0.6f_12+3
$$
​		得到得图像是一个超平面，无法解决线性不可分问题。

<img src="src/linearModel_withoutActFunc.jpg" alt="linearModel_withoutActFunc" style="zoom:50%;" />



​		加了激活函数的图像，使用sigmoid饱和函数激活：

<img src="src/linear_withActFunc.jpg" alt="linear_withActFunc" style="zoom: 50%;" />



使用的工具[^2]



### 2.2 一些常用的激活函数

### 2.2.1 梯度消失/梯度弥散的问题

​		sigmoid函数是一类**logistic函数，即不管输入是什么，得到的输出都在0到1之间。**
$$
sigmoid(x)=\frac{1}{1+e^{-x}}
$$
<img src="src/sigmoid.jpg" alt="sigmoid" style="zoom:67%;" />

sigmoid这样的函数通常被成为**非线性函数，因为我们不能用线性的项来描述它。**很多激活函数都是非线性或者线性函数的组合。**在求梯度时候，sigmoid的导数为：**
$$
\frac{\partial{sigmoid}}{\part{x}}=\frac{e^{-x}}{(e^{-x}+1)^2}
$$
<img src="src/partial_sigmoid.jpg" alt="partial_sigmoid" style="zoom:50%;" />

由此可见，**当梯度很大的时候，经过sigmoid函数后梯度将会变得接几乎为0。根据参数更新算法，梯度为0意味着不更新。**如果网络中有太多的这种情况则意味着网络只有细微的更新，网络就不会有多大的改善。

#### 2.2.2 ReLU及变种

​		**ReLU,** *** Rectified Linear Unit,整流线性单元。***
$$
ReLU=max(0,x)
$$
优点：ReLU延缓了梯度消失的问题，且由于其线性特点，训练快很多。

缺点：如果$x\leq0$，那么ReLU输出为0，梯度也为0，权重将不再更新，导致节点不再学习。

​		**Leaky ReLU**
$$
Leaky \ ReLU =max(0.1x,x)
$$


​	









[^1]: https://zhuanlan.zhihu.com/p/27609238

[^2]: https://www.geogebra.org/3d

