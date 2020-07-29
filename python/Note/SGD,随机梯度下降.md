# SGD,随机梯度下降

普通GD算法：
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