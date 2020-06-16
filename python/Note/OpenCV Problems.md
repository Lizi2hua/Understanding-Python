#  OpenCV Problems

## 1.色彩空间

### 1.1 HSV色彩空间

​		HSV，Hue Saturation Value的缩写。Hue表示色相，在OpenCV中取值范围**[0,179]**。Saturation表示饱和度，可以理解为在白色中对应颜色的浓度（颜色的纯度），在OpenCV中取值**[0,255]**。Value表示明度，可以理解为在黑色中对应颜色的浓度**[0，255]**。

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Opencv\hsv.png" alt="50" style="zoom:30%;" />

```python
import cv2
import  numpy as np
img=cv2.imread('hsv.png')
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h_channel_range=[np.amin(hsv[:,:,0]),np.amax(hsv[:,:,0])]
print(h_channel_range)
s_channel_range=[np.amin(hsv[:,:,1]),np.amax(hsv[:,:,1])]
print(s_channel_range)
v_channel_range=[np.amin(hsv[:,:,2]),np.amax(hsv[:,:,2])]
print(v_channel_range)
#[0,179][0,255][0,255]
```

​		OpenCV中有现成的函数cvtColor函数可以将RGB空间转换为HSV，并设置参数为COLOR_BGR2HSV那么所得的H、S、V值范围分别是[0,180)，[0,255)，[0,255)，而非[0,360]，[0,1]，[0,1]。

```python
img=cv2.imread("lane.jpg")
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
```



## 2.形态学操作

​		形态学操作就是基于形状的一系列图像处理操作。通过将**结构元素**（kernel）作用于输入图像来产生输出图像。

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\Morphology_1_Tutorial_Theory_Original_Image.png" alt="Morphology_1_Tutorial_Theory_Original_Image" style="zoom:80%;" />

### 2.1 膨胀（dilate）

​		此操作将图像**A**与任意形状的内核**B**，通常为正方形或圆形，进行卷积。内核**B**有一个可定义的**锚点**（anchor），通常定义为内核的中心点。

​		进行膨胀操作时，**将内核B滑过图像A，将内核B覆盖区域的*最大像素值*提取，并替代锚点位置的像素**。显然，最大值提取这一操作将会使上图的白色区域（255）变大，黑色区域（0）变小，即高亮区域会变大。

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\Morphology_1_Tutorial_Theory_Original_Image_j.png" alt="Morphology_1_Tutorial_Theory_Original_Image_j" style="zoom:80%;" />

```python
import cv2
#读入灰度图，0表示灰度图
img=cv2.imread("images/5.jpg",0)
#定义核,cv2.MORPH_RECT表示核形状为矩形（有椭圆，十字）
kernel=cv2.getcv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dst=cv2.dilate(img,kernel)
```



### 2.2 腐蚀（erode）

​		腐蚀与膨胀相对，不过它提取的是核覆盖下像素的**最小值**。进行膨胀操作时，**将内核B滑过图像A，将内核B覆盖区域的*最小像素值*提取，并替代锚点位置的像素**。显然，最大值提取这一操作将会使上图的黑色区域（255）变大，白色区域（0）变小，即高亮区域会变小。

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\Morphology_1_Tutorial_Theory_Erosion_erodedJ.png" alt="Morphology_1_Tutorial_Theory_Erosion_erodedJ" style="zoom:80%;" />

```python
import cv2
img=cv2.imread("images/5.jpg",0)
#定义核,cv2.MORPH_RECT表示核形状为矩形（有椭圆，十字）
kernel=cv2.getcv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dst=cv2.erode(img,kernel)
```

### 2.3 开运算（Opening)

​		开运算通过先对图片先腐蚀后膨胀（开）实现，能够排除小团块物体（噪声）[^1]
$$
dst=open(src,kernel)=dilate(erode(src,kernel))
$$

```python
import cv2
img=cv2.imread('4.jpg')
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dst=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
```

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Opencv\images\4.jpg" alt="4" style="zoom:80%;" />

​    

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\4_open.jpg" alt="4_open" style="zoom:80%;" />



### 2.4 闭运算（Closing)

​		闭运算与开运算相反，通过对图片先膨胀后腐蚀（闭）实现，能够补漏洞[^2]
$$
dst=close(src,kernel)=erode(dilate(src,kernel))
$$


<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Opencv\images\4.jpg" alt="4" style="zoom:80%;" />

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\4_close.jpg" alt="4_close" style="zoom:80%;" />

```python 
import cv2
img=cv2.imread('4.jpg')
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dst=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
```

### 2.5 顶帽（Top Hat)

​		原图像与开运算结果的之差,用来获取噪声[^3]。
$$
dst=tophat(src,kernel)=src-open(src,kernel)
$$

```python
import cv2
img=cv2.imread('4.jpg')
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dst=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
```

### 2.6 黑帽（Black Hat）

​		闭运算的结果与原图片之差，用来获取漏洞[^4]。
$$
dst=black(src,kernel)=close(src,kernel)-src
$$



## 3. 进一步认识卷积（Convolution）[^5]

​		卷积操作用于提取图片的特征，这些特征包括颜色特征，纹理特征，形状特征和空间关系特征。下图的卷积核为一个图片x轴方向的梯度算子（Sobel的Gx算子）

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\cnn_gif.gif" alt="cnn_gif" style="zoom:80%;" />

### 3.1 颜色特征

​		颜色特征是一种**全局**特征，描述**图像或图像区域**所对应物体的表面特征。一般颜色特征是基于像素点的特征，所有的像素点都对颜色特征有贡献。由于颜色对图像或图像区域的方向、大小等变化不敏感，所以**颜色特征不能很好地捕捉图像中对象的局部特征**。

### 3.2 纹理特征

​		纹理特征也是一种**全局**特征，它也描述了图像或图像区域所对应景物的表面性质。但由于纹理只是一种物体表面的特性，并不能完全反映出物体的本质属性，所以仅仅利用纹理特征是无法获得高层次图像内容的。与颜色特征不同，纹理特征不是基于像素点的特征，它需要在包含**多个像素点的区域中进行统计计算**。

### 3.3 形状特征

​		形状特征仅描述了目标的**局部**的性质

（待补充）

### 3.4 空间关系特征

（待补充）



## 4. 滤波

### 4.1 双边滤波（Bilateral Filter）

​		由于高斯滤波在边缘区域会模糊边缘，因此滤波结果会丢失边缘信息。在高斯滤波的基础上，双边滤波叠加了像素值的考虑，因此滤波效果对保留边缘更有效。

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\Gaussian_Fliter.png" alt="Gaussian_Fliter" style="zoom:80%;" />

​		双边滤波的核函数是**空间域核**与**像素范围域核**的综合结果：

1. ​	在图像的平坦区域，像素值变化很小，那么像素差值接近于0，对应的像素范围域权重接近于1，此时空间域权重起主要作用，相当于进行高斯模糊；

2. 在图像的边缘区域，像素值变化很大，那么像素差值大，对应的像素范围域权重变大，即使距离远空间域权重小，加上像素域权重总的系数也较大，从而保护了边缘的信息。

   <img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\bialteral_Filter_.png" alt="bialteral_Filter_" style="zoom: 67%;" />

​	

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\743748-20200321155209109-477384372.png" alt="743748-20200321155209109-477384372" style="zoom:75%;" />



~~~python
import cv2
img=cv2.imread('5.jpg')
dst=cv2.bilteralFilter(img,11,75,75)
~~~

### 4.2 Sobel算子

​		Sobel算子是一个离散的一阶差分算子，主要用于计算图像**亮度（0-255）函数的一阶梯度进似值**。对图像任一像素使用sobel算子，都会产生该像素点对应的梯度矢量[^6]。
$$
G_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}*\begin{bmatrix}1&2&255\\1&2&255\\1&2&255\end{bmatrix}
$$
​		Sobel算子有两个，一个用于检测水平边缘，一个用于检测垂直边缘。将这个两个矩阵与图像做卷积运算，得到横向、纵向的亮度差分近似值Gx,Gy。
$$
G_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}
G_y=\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}
$$
​		梯度辐值是Gx和Gy的绝对值之和。

​		Scharr算子是对Sobel算子的改进（数值变大，1替换为3，2替换为10）

~~~python
import cv2
img=cv2.imread("2.jpg")
sobel_x=cv2.Sobel(img,-1,1,0)
sobel_y=cv2.Sobel(img,-1,0,1)

~~~

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\1592293845(1).jpg" alt="1592293845(1)" style="zoom:80%;" />

------

### 4.3 Canny算法

​		Canny算法是一个多级边缘检测（edge detection）算法，通常情况下边缘检测的目的是在保留原有图像属性的情况下，显著减少图像的数据规模。canny的目标是找到一个最优边缘检测算法，即：

- 最优检测：算法能够能可能多地标识出图像中的实际边缘，漏检率和误检率都尽可能小；
- 最优定位：检测到的边缘点的位置距离实际边缘点的位置最近，或者是由于噪声影响引起检测出的边缘偏离物体的真实边缘的程度最小；
- 检测点与边缘点一一对应：算子检测的边缘点与实际边缘点应该是一一对应。

算法实现步骤：

1. 使用高斯滤波平滑图像，去除噪声
2. 寻找图像像素的强度梯度（sobel）
3. 用非极大值抑制（NMS）计数来消除误检，达到瘦边效果。
4. 用双阈值的方法来决定可能的边界
5. 用滞后计数来跟踪边界

​         双阈值，设定一个阈值上界和阈值下界（opencv中通常由**人为指定**的），图像中的像素点如果大于阈值上界则认为必然是边界（称为强边界，strong edge），小于阈值下界则认为必然不是边界，两者之间的则认为是候选项（称为弱边界，weak edge），需进行进一步处理。滞后技术认为和强边界相连的弱边界是边界，其他的弱边界则被抑制。如果该色块的值贯穿最大值于最小值，那么就不是边缘，舍弃。

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\1592299891(1).jpg" alt="1592299891(1)" style="zoom:50%;" />

~~~python
import cv2

img=cv2.imread('images/25.jpg',0)
cv2.imshow('origin',img)

img=cv2.convertScaleAbs(img,alpha=10,beta=10)
cv2.imshow('abs',img)
img=cv2.GaussianBlur(img,(3,3),10)
cv2.imshow('gauss',img)
canny=cv2.Canny(img,100,150)
cv2.imshow('canny',canny)
~~~







## 																		脚注

[^1]: 噪声一般都是异常值（outlier），异常值指一组测定值中与平均值的偏差超过两倍标准差的测定值，所以经过先腐蚀后膨胀操作能够去掉噪声（腐蚀作用），并尽可能保留原图的形状（膨胀）。（*自己的想法*）
[^2]: 漏洞一般是形状比较小的值为零的区域（对黑色背景而言）,膨胀使图片的亮度区域变大，将漏洞填充，但是整个形状会变胖，所以腐蚀之后能尽可能保留原图形状。（*自己想法*）
[^3]: 开运算能够去除噪声，有噪声图减去无噪声图得到噪声图。
[^4]: 闭运算能够修补漏洞，闭运算结果减去有漏洞的原图得到漏洞。

[^5]: CNN中使用卷积神经网络来提取图片的特征，随着卷积层的加深，特征图也越抽象，维度越小。最后一次卷积得到的特征图作为后面全连接网络的先验，体现了贝叶斯的思想。（*自己的想法*）
[^6]: 梯度算子这么设计类似于求微分（x轴方向的[-1,0,1]为例），即{f(x+Δx)-f(x)}/Δx



------

## 																	  借物表

[OpenCV教程]：https://www.kancloud.cn/aollo/aolloopencv/263731

[膨胀与腐蚀]：http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html

[CNN可视化]https://poloclub.github.io/cnn-explainer/

[Canny算法]：[https://baike.baidu.com/item/canny%E7%AE%97%E6%B3%95/8439208?fr=aladdin](https://baike.baidu.com/item/canny算法/8439208?fr=aladdin)