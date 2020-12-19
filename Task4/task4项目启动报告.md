# Task4 项目启动报告

## 项目背景
我们小组的项目主题是解决基于事件相机数据的步态识别问题。
### 事件相机
#### 简介

动态视觉传感器 (Dynamic Vision Sensors,DVS) 是一种受生物视觉原理的启发而产生的传感器，用于视觉信息的捕捉，又被称为事件相机 (Event Camera) 。事件相机可以捕捉到微秒级别的亮度变化以及拥有高达 140 dB 的动态范围，能够适应很多复杂苛刻的光照环境和高速场景。他在 SLAM 、机器人视觉、物体识别等众多领域与传统 RGB 相机相比，具有巨大的优势。
#### 运行机制
事件相机的运行模式与传统相机相比有很大的不同。相比于传统 RGB 相机以固定的速率捕捉像素，产生一帧一帧 (frame) 的图片，事件相机采用另一套完全不同的感知模式。事件相机捕捉所有像素独立的亮度变化，并将其中超过阈值的亮度变化异步变化处理为一个个事件。这些事件随时间产生，总体我们称其为事件流 (event streams) 。

具体来说，对于事件相机，不同位置事件发生的是异步的，每一个事件可以被描述为一个四元组: ![](https://latex.codecogs.com/png.latex?(t,x,y,p)) ，其中 ![](https://latex.codecogs.com/png.latex?t) 是事件发生的时间戳， ![](https://latex.codecogs.com/png.latex?(x,y)) 是事件发生时在二维像素空间中的位置，而 ![](https://latex.codecogs.com/png.latex?p) 用来表示亮度的变化情况，其取值只有‘增加’或者‘减少’，常用 ![](https://latex.codecogs.com/png.latex?p=+1) 来表示表示像素强度的增加， ![](https://latex.codecogs.com/png.latex?p=-1) 表示减少。在事件相机的运行过程中，传感器仅在某一像素的强度变化超过了一定的阈值，也就是说满足:

<div align=center>

![](https://latex.codecogs.com/png.latex?\large&space;\log&space;I_{now}^{x,y}-\log&space;I_{previous}^{x,y}&space;>&space;\theta)
</div>

其中 ![](https://latex.codecogs.com/png.latex?I_{now}^{x,y}) 和 ![](https://latex.codecogs.com/png.latex?I_{previous}^{x,y}) 是在同一像素 ![](https://latex.codecogs.com/png.latex?(x,y)) 的位置上当前和之前的光照强度， ![](https://latex.codecogs.com/png.latex?\theta) 为强度阈值。

事件相机的传感器的设计与传统的 RGB 摄像机相比有许多独特的优势。首先，事件相机需要的资源更少。由于事件只有在检测到达到阈值的强度变化时才会触发，相比于传统 RGB 相机，产生的事件其实是稀疏的，数据量与能耗都很低。其次，事件相机的时间分辨率为几十微秒，在高速运动下能够捕捉到非常详细的数据，而不会出现模糊或滚动快门问题。最后一点，事件相机具有拥有高达 140dB 的动态范围，相比于传统 RGB 相机 60dB 的动态范围，事件相机能在更多的环境下工作。

虽然事件相机具有非常大的应用潜力，然而由于其事件流的数据特征，传统的计算机视觉与机器学习算法无法直接用于其图像识别领域。这极大的限制了事件相机的发展前景。

### 步态识别

因此在本实验中，我们提出了一些想法来使用事件相机数据进行经典的步态识别问题。具体来说。步态识别问题的目的是根据人的行走视频来确定人的身份。步态识别问题是非常经典的计算机视觉的问题，拥有广泛的应用，如疑犯追踪、数字医疗与安全监控等情景。

<img src="https://pic.downk.cc/item/5fddc8413ffa7d37b384360a.jpg" width="400">
在本次实验中，我们将探究一种基于事件的步态识别技术(EV-Gait)，使他能够处理嘈杂光照环境下的事件流，实现数据降噪，并根据事件流准确推断出步态的身份。


## 项目目标
使用 Microsoft NNI(Neural Network Intelligence) 优化 EV-Gait ，从而进一步提升 EV-Gait 的识别准确率。

## 项目规划
目前有2套设想：
### 设想一
使用 NNI 定义好并推荐使用的功能，调优 EV-Gait ，包括但不限于以下方面：
- 调优超参：使用 NNI 的 HPO 功能，自动调优 EV-Gait 神经网络传统的超参数，如 learning rate 、 batch size 、 momentum 、 dropout rate 等。
- 微调网络：使用 NNI 微调网络，如调整 hidden layer 的大小，卷积核的大小等。
- 调优网络架构：使用 NNI 的 NAS 功能，对网络结构做较大调整。
### 设想二

方案二具体方案仍处于构思阶段，有待进一步学习研究

我们打算使用 NNI 调优本实验中特别的、非传统的超参数 ![](https://latex.codecogs.com/png.latex?n) ：将两张连续的 frame 拍摄下来的时间间隔内发生的全部 events pack 成类似一张图片的表示后，再选择将 ![](https://latex.codecogs.com/png.latex?n) 张连续的 frame 再打包为一组。

这将会从根本上改变我们输入进网络进行训练的数据形式，从而要求我们将神经网络的结构从传统的 CNN 修改为 LSTM 。

设想一的详细方案请见实施方案部分。

## 实施方案
<img src="https://pic.downk.cc/item/5fddcdfb3ffa7d37b388c09e.jpg">

### 数据处理
在 Event Camera 采集 event 数据时，会得到一连串的 event 流,每个 event 以四元组 ![](https://latex.codecogs.com/png.latex?(t,x,y,p)) 存储记录,其中 ![](https://latex.codecogs.com/png.latex?t) 为 event 发生的时间戳， ![](https://latex.codecogs.com/png.latex?x) 、 ![](https://latex.codecogs.com/png.latex?y)  是 event 在2D 像素空间中发生的位置坐标， ![](https://latex.codecogs.com/png.latex?p)  是 event 的极性。
与此同时同时存在一个传统 RGB 相机，采集一帧帧连续的 frame ，我们也会记录 frame 产生的时间戳。
然而， Event Camera 收集到的 event 流与传统图像差别很大，无法被直接应用到最先进的 CNN 中。为了解决该问题，我们采用了将 events 转变为类似图片的表示的办法，具体方法将在不久后详细介绍。

传统的 RGB 相机可以记录下一帧帧连续的 frame ，然而，这些 frame 之间仍存在较大的时间间隔，因此这些连续的 frame 产生时间戳可以将时间轴划分成多个时间区间。简单起见，后面我们将两张连续的 frame 之间的时间区间简称为时间区间。

正如前面提及，Event Camera 可以捕捉精确到微秒级别的亮度变化，其产生的频率远远大于 frame ,因此在每个时间区间内，实际上会发生许多个 event ，具体来说可能有上万个，被事件照相机所采集。

在我们用于训练与测试的数据集 DVS128_2020 中，有着许多 .txt 后缀的文件，每个 .txt 文件内记录着上万个 event 的数据，这些 event 之间并非毫无关联。实际上，DVS128_2020 数据集将哪些 event 打包为一个 .txt 文件的参考依据是：这些 event 的时间戳是否处于同一个时间区间内。也就是说，每个 .txt 文件内存储的同一个时间区间内发生的全部 event 。这样 .txt 文件就与时间区间产生了一一对应的关系。

DVS128_2020 数据集已经按照上述规则将 event 打包起来，之后我们需要自行将每组打包起来的 events 转变为类似一张图片的表示，称之为 "event image" 。传统的 RGB 图像有3个R、G、B3个 channel ，而 event image 具有4个 channel ,前两个 channel 分别记录在一个像素点上 positive event 发生的次数与 negative event 发生的次数。后两个 channel 分别记录着 positive event 和 negative event 的 ratio ，它描述了时间特征。在像素点 ![](https://latex.codecogs.com/png.latex?(i,j))  上的ratio ![](https://latex.codecogs.com/png.latex?r_{i,j}) 定义为：

<div align=center>

![](https://latex.codecogs.com/png.latex?\large&space;r_{i,j}=\frac{t_{i,j}-t_{begin}}{t_{end}-t_{begin}})
</div>

其中 ![](https://latex.codecogs.com/png.latex?t_{i,j}) 是在 pixel![](https://latex.codecogs.com/png.latex?(i,j))上 发生的最后一个 positive 或 negative 事件的时间戳。 ![](https://latex.codecogs.com/png.latex?t_{begin}) 与 ![](https://latex.codecogs.com/png.latex?t_{end}) 分别为相对整个事件流而言发生的第一个事件的时间戳与发生的最后一个事件的时间戳。这两个 ratios 评估了在不同位置物体的生命周期。

经过上述步骤后， 被打包的多组 events 被表示成了多张一一对应的 event images 。因为 event image 与 RGB 图片有着极为相似的结构，我们就可以将 event image 送进CNN进行深度学习了。
### 网络结构

本实验中，初步构建的基于事件的步态识别的深度神经网络的结构如下图所示：
<img src="https://pic.downk.cc/item/5fddcda73ffa7d37b3888504.png">

整个网络大致可以分为两个主要部分：先通过卷积层与残差块（ResBlock）层进行特征提取，然后通过全连接层与 softmax 对特征进行识别，并推断结果。卷积层已经被证明是一种有效特征提取方法，并被广泛应用于图像分类中。而ResBlock层能一定程度上解决特征消失问题，并使得网络的深度增加，以便卷积层提取的特征可以更好地整合。特征经过全连接层的整合与关联，被传入 softmax 函数进行分类。
具体的参数，输入的图像依次通过卷积核大小为 ![](https://latex.codecogs.com/png.latex?\large&space;7*7)、![](https://latex.codecogs.com/png.latex?\large&space;3*3)、![](https://latex.codecogs.com/png.latex?\large&space;3*3)、![](https://latex.codecogs.com/png.latex?\large&space;3*3) 的四层卷积层，其通道数分别为 ![](https://latex.codecogs.com/png.latex?\large&space;64)、![](https://latex.codecogs.com/png.latex?\large&space;128)、![](https://latex.codecogs.com/png.latex?\large&space;256) 和 ![](https://latex.codecogs.com/png.latex?\large&space;512)。接着通过两个ResBlock层，其卷积核大小都为 ![](https://latex.codecogs.com/png.latex?\large&space;3*3) ，通道数为 ![](https://latex.codecogs.com/png.latex?\large&space;512) 。接下来两个全连接层的大小为 ![](https://latex.codecogs.com/png.latex?\large&space;1024) 和 ![](https://latex.codecogs.com/png.latex?\large&space;512) 个节点，最后经过 softmax 函数，输出识别结果。

### 实验计划

#### 代码开发
- 数据处理
- 实现网络
- 训练可视化
- 嵌入NNI 

#### 实验
- 使用NNI优化网络
- 使用NNI参数调优
- 结果评估



