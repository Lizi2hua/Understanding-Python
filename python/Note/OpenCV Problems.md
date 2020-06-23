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

2. ​    在图像的边缘区域，像素值变化很大，那么像素差值大，对应的像素范围域权重变大，即使距离远空间域权重小，加上像素域权重总的系数也较大，从而保护了边缘的信息。

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



## 5. 图像金字塔

​		图像金字塔是图像处理和计算机视觉中常用的概念，常用于**多尺度图像处理（multiscale processing)**领域。早年的图像匹配、识别等算法中都用到了图像金字塔。

### 5.1 高斯金字塔

​		高斯金字塔是一种下采样方式，如下图所示。底层为原图，每向上一层则是通过高斯滤波和1/2采样，即去掉偶数列和行。高斯金子塔也能进行上采样（即放大操作）

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\guassian_prymid.png" alt="guassian_prymid" style="zoom:80%;" />

```python
import cv2
img=cv2.imread('images/1.jpg')
for i in range(5):
    cv2.imshow(f'img{i}',img)
    # img=cv2.pyrUp(img) 放大，会变得越来越模糊
    img=cv2.pyrDown(img) #缩小
cv2.waitKey(0)

```



### 5.2 拉普拉斯金字塔

​		在进行高斯金字塔运算时，由于补断进行高斯滤波和下采样，丢失了许多高频信号（边缘信息）。拉普拉斯金字塔的目的就是保存这些高频信号，**保存这些高频信号的方式是保存差分图**。拉普拉斯金字塔由高斯金字塔计算得来。
$$
L_i=G_i-PryUp(Gi+1)
$$

~~~python	
import cv2
img=cv2.imread('images/1.jpg')
img_down=cv2.pyrDown(img)
img_up=cv2.pyrUp(img_down)
# 这一步图片变得模糊
cv2.imshow('0',img)
cv2.imshow('1',img_up)
img_new=cv2.subtract(img,img_up)
# 这一步得到边缘图像
cv2.imshow('2',img_new)
cv2.waitKey(0)

~~~

## 6. 霍夫（Hough）变换

​		直线霍夫变换通过将笛卡尔坐标系中的直线表达式：
$$
y=kx+b
$$
​		转换到参数空间：
$$
b=-xk+y
$$
​		如下图所示，霍夫空间的一条直线可在笛卡尔坐标中唯一确定一个点，反之，在笛卡尔坐标系中的一个点，在霍夫空间中由无数条直线。判断**三个点**是否在同一条直线上，只需判断在霍夫空间中三个点的确定的直线是否相交。

​		但是，***如果直线接近竖直方向，则会由于k的值都接近无穷而使得计算量增大***，此时我们使用直线的极坐标方程来表示直线。在使用Hough变换前一般要进行Canny操作。
$$
\rho=xcos\theta+ysin\theta
$$


![hough_trans](C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\hough_trans.jpg)



<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\v2-8de2df486bdb0c2fc81a8714496207a7_r.jpg" alt="v2-8de2df486bdb0c2fc81a8714496207a7_r" style="zoom:40%;" />

```python
import  cv2
import numpy as np

img=cv2.imread("images/2.jpg")

gauss=cv2.GaussianBlur(img,(3,3),0)
canny=cv2.Canny(gauss,100,200)

# lines=cv2.HoughLinesP()是对HoughLines的改进
lines=cv2.HoughLines(canny,1,np.pi/180,40)
# 返回值 lines 中的每个元素都是一对浮点数，表示检测到的直线的参数，即（r,θ），是 numpy.ndarray 类型。

```



## 7. 分水岭算法

​		可以将图像中每一个像素点的灰度值看作该点的海拔高度。模拟泛洪算法的基本思想是：假设在**每个区域最小值**的位置上打一个洞并且让水以均匀的上升速率从洞中涌出（腐蚀操作），从低到高淹没整个地形。*当处在不同的汇聚盆地中的水将要聚合在一起时，修建的大坝将阻止聚合。*水将达到在水线上只能见到各个水坝的顶部这样一个程度。**这些大坝的边界对应于分水岭的分割线。所以，它们是由分水岭算法提取出来的(连续的)边界线。** 

​		如下图，原图像（灰度图）根据像素值大小转换成的梯度图。

![water_shed_alogrithm1](C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\water_shed_alogrithm1.gif)

<img src="C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\地形图.png" alt="地形图" style="zoom:75%;" />

​		算法过程如下图所示。

![water_shed_alogrithm](C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\water_shed_alogrithm.gif)

​		**分水岭算法对噪声的影响非常敏感**，所以在真实图像中，由于噪声点或者其它干扰因素的存在，使用分水岭算法常常存在过度分割的现象，这是因为很多很小的局部极值点的存在。

​		**可以通过融入预处理步骤来限制允许存在的区域的数目**。通过引入**标记**来控制过度分割。一个标记是属于一幅图像的一个连通分量。与**感兴趣物体**相联系的标记成为**内部标记**，与**背景相关联**的标记称为**外部标记**。

​		通常的标记图像，都是**在某个区域定义了一些灰度层级**，在这个区域的洪水淹没过程中，**水平面都是从定义的高度开始的**，这样可以避免一些很小的噪声极值区域的分割。

![water_shed_with_marker](C:\Users\Administrator\Desktop\Project：777\CODE\python\Note\src\water_shed_with_marker.gif)



## 8. 平均池化（Average Pooling)

​	将图片按照固定大小的网格分割，网格内像素的平均值代表这个网格的值。我们将这种把图片使用均等大小网格分割，并求网格内代表值的操作称为**池化（Pooling）**。

​	实现代码如下：

```python
import numpy as  np
import cv2
import glob

# 读图
img_path=glob.glob(r"../Opencv/images/*.jpg")
grid_size=(8,8)

for i in range(len(img_path)):
    img=cv2.imread(img_path[i])
    # 平均池化
    h,w,c=img.shape
    g_w,g_h=grid_size
    # 算出需要多少个网格来覆盖图片
    h_nums=int(h/g_h)
    w_nums=int(w/g_w)
    # 图片是3d的，对每一层做池化
    for i in range(h_nums):
        for j in range(w_nums):
            for k in range(c):
                img[i*g_h:(i+1)*g_h,j*g_w:(j+1)*g_w,k]=np.mean(img[i*g_h:(i+1)*g_h,j*g_w:(j+1)*g_w,k]).astype(np.uint8)

    # 图片的w,h
    # 缺陷：不能整除g_w,g_h的情况下处理?
    cv2.imshow('pic',img)
    cv2.waitKey(1500)
```









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

[图像金字塔]https://www.jianshu.com/p/e3570a9216a6

[分水岭算法]https://blog.csdn.net/Lemon_jay/article/details/89355937