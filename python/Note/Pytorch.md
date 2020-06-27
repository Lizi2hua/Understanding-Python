Pytorch

## 1.Dataset和Dataloader[^1]

​		pytorch的数据加载到模型的操作顺序如下:

​		1.创建一个`Dataset`对象

​		2.创建一个`DataLoader`对象

​		3.循环这个`DataLoader`对象,将data,label加载到模型中训练

### 1.1 Dataset

​		Datasets是一个pytorch定义的dataset的源码集合。所有其他*数据集都应该进行子类化*，所有的子类**都应该override`__len__`和`__getitem__`**。前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)，初始化放在`__init__`。

```python
from torch.utils.data import Dataset
class CustomDataset(Dataset):#需要继承data.Dataset
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, 			PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
		return 0
```

pytorch中定义y一个数据集

~~~python
class data_set(data.Dataset):
    # must inherit Dataset class
    def __init__(self,img_path,label_path,transform=None):
        # define path
        # """img_path should be like:path=glob.glob("..\mchar_test_a\*.png")"""
        # tansform参数
        self.img_path=img_path
        self.label_path=label_path
        if transform is not None:
            self.transform=transform
        else:
            self.transform=None


    def __getitem__(self, item):
        # get data
        # item是默认的，用于单个读取时的迭代【个人猜想】
        # data
        imgs=Image.open(self.img_path[item])
        if self.transform is not None:
            imgs=self.transform(imgs)
        # labels
        # 最好传入Dataset类的是已经处理好的label的list，否则容易报错
        # 5位数（数据集内最高位为6位）定长字符字符串，比如23则是[2,3,10,10,10]
        lbl = np.array(self.label_path[item], dtype=np.int)
        # lbl=np.array(self.label_path[item],dtype=np.int)
        lbl=list(lbl)+(5-len(lbl))*[10]#[2,3]+3*[10]
        return imgs,torch.from_numpy(np.array(lbl[:5]))


    def __len__(self):
        # 定义数据集的大小
        return len(self.img_path)
~~~

​	验证是否成功:

```python
"""定义成功了嘛？"""
if __name__ == '__main__':

    img_path = glob.glob("..\datasets\street_signs\mchar_train\mchar_train\*.png")
    # 最好传入Dataset类的是已经处理好的label，否则容易报错
    label_path = "..\datasets\street_signs\mchar_train.json"
    label_path=json.load(open(label_path))
    label_path=[label_path[x]['label'] for x in label_path]
    print(len(img_path),len(label_path))

    train_loader = torch.utils.data.DataLoader(
        data_set(img_path, label_path,
                 transforms.Compose([
                     transforms.Resize((64, 128)),
                     transforms.RandomCrop((60, 120)),
                     transforms.ColorJitter(0.3, 0.3, 0.2),
                     transforms.RandomRotation(5),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ])),
        batch_size=40,
        shuffle=True,
        num_workers=10,
    )

    for inputs,labels in train_loader:
        print(inputs)
        print(labels)
        break
```



### 1.2 glob.glob的用法[^2]

​		可以使用`..\mchar_test_a\*.png`这样的语法，返回该文件夹下所有以`.png`结尾的文件的名字，并以`list`存储.

```python
import glob
path=glob.glob("..\datasets\street_signs\mchar_test_a\mchar_test_a\*.png")
print(type(path))
#输出 <class 'list'>
```

### 1.3 字典遍历

​		由于字典属于可迭代对象[^3],迭代返回的值是键值.

```python
dic={'key1':1,'key2':2,'key3':3,'key4':4}
print([x for x in dic])
-->['key1', 'key2', 'key3', 'key4']

```

### 1.4 torch.utils.data.DataLoader[^4]

​		DataLoader是pytorch的重要接口,它的目的是:将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练。

​		DataLoader的参数:

- **dataset(Dataset)**: 传入的数据集
- **batch_size(int, 可选)**: 每个batch有多少个样本
- **shuffle(bool, 可选)**: 在每个epoch开始的时候，对数据进行重新排序
- **sampler(Sampler, 可选)**: 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
- **batch_sampler(Sampler, 可选)**: 与sampler类似，但是一次只返回一个batch的indices（索引）.需要注意的是，一旦指定了这个参数，那么batch_size,shuffle,sampler,drop_last就不能再制定了（互斥——Mutually exclusive）
- **num_workers (int, 可选)**: 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0)
- **collate_fn (callable, 可选)**: 将一个list的sample组成一个mini-batch的函数
- **pin_memory (bool, 可选)**： 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.
- **drop_last (bool, 可选)**: 如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了.
- **timeout(numeric, 可选)**: 如果是正数，表明等待从worker进程中收集一个batch等待的时间，若超出设定的时间还没有收集到，那就不收集这个内容了。这个numeric应总是大于等于0。默认为0.
- **worker_init_fn (callable, 可选)**: 每个worker初始化函数.



### 1.5 pipeline概念[^5]

​		以一个餐厅为例，将一件需要重复做的事情（为客人准备一份精美的食物）切割成各个不同的阶段（准备盘子，准备薯条，准备豌豆，准备饮料），每个阶段由独立的单元负责。。所有待执行的对象依次进入作业队列（这里是所有的客户排好队依次进入服务，除了开始和结尾的一段时间，任意时刻，四个客户被同时服务）。对应到CPU中，每一条指令的执行过程可以切割成：fetch instruction、decode it、find operand、perform action、store result 5个阶段。

### 

## 2. torchvison

### 2.1 torchvision.transforms[^6]

​		Transforms是一些常见的图像变换，（这些变换的结果）它们可以通过`Compose`方法连接（chain togther）到一起。此外还有`torchvision.transform.functional `模块，可以通过该模块实现对变换的精确控制。当在需要建立一个更复杂的pipeline时很用。

源代码分析

- `torchvision.transforms.Compose(*transforms*)`

  ```python
  >>> transforms.Compose([
  >>>     transforms.CenterCrop(10),
  >>>     transforms.ToTensor(),
  >>> ])
  ```

- `torchvision.transforms.ToTensor`

  ```python
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    一般情况下，将一个PIL Image 或者 numpy.ndarray(H*W*C)从[0,255]范围内转换为torch.FloatTensor(C*W*H)，范围为[0.,1.]"""
  ```

- `torchvision.transforms.ToPILImage`

  ```python
  """将Tensor转换成PIL Image"""
  ```

- `torchvision.transforms.Normalize`,根据`__init__`定义需要`std`和`mean`两个参数。根据一下公式进行归一化：
  $$
  val_c=(val_c-mean_c)/std_c
  $$
  val,mean,std皆为3维list。

  ```python
  """Normalize类是用于做数据归一化，一般都会对数据进行这样的操作
  """
      def __init__(self, mean, std, inplace=False):
          self.mean = mean
          self.std = std
          self.inplace = inplace
  
  ```

- `torchvision.transforms.Resize`

  ```python
  """对PIL Image作Resize操作，使之变为指定size"""
     def __init__(self, size, interpolation=Image.BILINEAR):
          assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
          self.size = size
          self.interpolation = interpolation
  ```

- `torchvision.transforms.RandomCrop`

- `torchvision.transforms.RadomHorizontalFlip `随机图像水平翻转

- `torchvision.transforms.RadomVerticalFlip `随机图像竖直翻转

- `torchvision.transforms.ColorJitter`

  ```python
   """Randomly change the brightness(亮度), contrast（对比度） and saturation（饱和度） of an image."""
  ```

- `torchvision.transforms.RandomRatation`随机旋转
- `torchvision.transforms.Grayscale`转成灰度图



### 2. 2 torchvision.models[^7]

​	该模型子包包含多种面向不同任务的模型，包括：图片分类，像素级的语义分割，目标检测，实例分割，人的关键点检测以及视频分类。并且，pytorch还提供了与训练的模型。

​	**分类任务**：

- AlexNet
- VGG

- ResNet

- SqueezeNet

- DenseNet

- Inception v3

- GoogLeNet

- ShuffleNet v2

- MobileNet v2

- ResNeXt

- Wide ResNet

- MNASNet

构建模型并使用预训练的模型

```python
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True
```

可以用一下代码来查看网络结构参数：

```python
import torchvision.models as models
resnet=models.resnet18()
print(resnet)
print('___________')
print(list(resnet.children())[:-1])
```



**语义分割**

- FCN ResNet50, ResNet101
- DeepLabV3 ResNet50, ResNet101

**目标检测**

- Faster R-CNN ResNet-50 FPN
- Mask R-CNN ResNet-50 FPN
- Keypoint R-CNN

**视频分类**

- ResNet 3D 18
- ResNet MC 18
- ResNet (2+1)D








------

[^1]: https://blog.csdn.net/guyuealian/article/details/88343924
[^2]: https://blog.csdn.net/csapr1987/article/details/7469769
[^3]: https://blog.csdn.net/LaoYuanPython/article/details/89609187
[^4]: https://blog.csdn.net/g11d111/article/details/81504637
[^6]: https://pytorch.org/docs/master/torchvision/transforms.html
[^5]: https://www.cnblogs.com/midhillzhou/p/5588958.html

[^7]: [https://pytorch.org/docs/stable/torchvision/models.html?highlight=models%20resnet18#torchvision.models.resnet18](https://pytorch.org/docs/stable/torchvision/models.html?highlight=models resnet18#torchvision.models.resnet18)

