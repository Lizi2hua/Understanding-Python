# Faster RCNN

## 1.介绍

​		一些目标检测网络需要**区域提案（Region Proposal）**算法来**假设**目标的位置，着暴露了区域提案是一个瓶颈。本论文介绍了了名为**RPN,区域提案网络**的方法，该网络与detection network共享全图像的卷积特征，着几乎cost free。

## 2. RPN

​		A RPN takes an image (of any size) as input and outputs **a set of rectangle object proposals**,每个都 proposal 都有一个对象得分。

​		