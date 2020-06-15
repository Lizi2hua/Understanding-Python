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




------

## 																		脚注

[^1]: 噪声一般都是异常值（outlier），异常值指一组测定值中与平均值的偏差超过两倍标准差的测定值，所以经过先腐蚀后膨胀操作能够去掉噪声（腐蚀作用），并尽可能保留原图的形状（膨胀）。（*自己的想法*）
[^2]: 漏洞一般是形状比较小的值为零的区域（对黑色背景而言）,膨胀使图片的亮度区域变大，将漏洞填充，但是整个形状会变胖，所以腐蚀之后能尽可能保留原图形状。（*自己想法*）
[^3]: 开运算能够去除噪声，有噪声图减去无噪声图得到噪声图。
[^4]: 闭运算能够修补漏洞，闭运算结果减去有漏洞的原图得到漏洞。







------

## 																	  借物表

[OpenCV教程]：https://www.kancloud.cn/aollo/aolloopencv/263731

[膨胀与腐蚀]：http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html