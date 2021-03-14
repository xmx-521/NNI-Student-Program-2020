# Task 2.2 进阶任务项目说明文档

——基于自动机器学习工具 NNI 的创新性科研扩展项目说明

---

## 团队基本信息

- 团队名：电脑一带五
- 团队成员：徐满心、李乐天、孟繁青、尚丙奇、陈垲昕
- 团队学校：同济大学
- Github Repo地址: https://github.com/xmx-521/NNI-Student-Program-2020

## 任务要求：NNI-task2.2 进阶任务

### 文档情况

- [x] 应用nni对于CIFAR10的Task2.1示例进行超参调优以及神经网络架构搜索

### 文档内容描述

本文档为CIFAR10图像分类样例分析报告，具体内容包括

- 超参调优

- TPE简介

- 神经网络架构搜索

- 应用nni

- 实验过程总结

  

## 1.自动超参调优

自动调优是 NNI 提供的关键功能之一，主要应用场景是 超参调优。 应用于 Trial 代码的调优。 提供了很多流行的 自动调优算法（称为 Tuner ）和一些提前终止算法（称为 Assessor）。 NNI 支持在各种培训平台上运行 Trial，例如，在本地计算机上运行， 在多台服务器上分布式运行，或在 OpenPAI，Kubernetes 等平台上。

NNI 的其它重要功能，例如模型压缩，特征工程，也可以进一步 通过自动调优来提高，这会在介绍具体功能时提及。

NNI 具有高扩展性，高级用户可以定制自己的 Tuner、 Assessor，以及训练平台 来适应不同的需求。



#### 1.1用法

为了让机器学习/深度学习模型适应不同的任务/问题，超参数总是需要调优。 自动化超参数调优的过程需要好的调优算法。NNI是使用其内置的tuner进行自动调优的，并且提供了先进的调优算法，使用上也很简单。 

我们先主要介绍我们所使用的调优算法 TPE


## 2.TPE简介

Tree-structured Parzen Estimator (TPE) 是一种 sequential model-based optimization（SMBO，即基于序列模型优化）的方法。

贝叶斯优化属于一类被称为*sequential model-based optimization*(SMBO)的优化算法。这些算法使用先前对损失 f 的观测，来确定下一个(最佳)点来取样 f。该算法大致可以概括如下。

1. 使用先前计算过的点 **X1**: n，计算损失 f 的后验期望值。
2. 在一个新的点 **X**new取样损失 f ，它最大化了 f 的期望的某些效用函数。该函数指定 f 域的哪些区域是最适合采样的。
[![609DsI.png](https://s3.ax1x.com/2021/03/14/609DsI.png)](https://imgtu.com/i/609DsI)

![[公式]](https://www.zhihu.com/equation?tex=f): 就是那个所谓的黑盒子，即输入一组超参数，得到一个输出值。

![[公式]](https://www.zhihu.com/equation?tex=X):是超参数搜索空间等。

![[公式]](https://www.zhihu.com/equation?tex=D):表示一个由若干对数据组成的数据集，每一对数组表示为![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29)，![[公式]](https://www.zhihu.com/equation?tex=x)是一组超参数,![[公式]](https://www.zhihu.com/equation?tex=y)表示该组超参数对应的结果。

![[公式]](https://www.zhihu.com/equation?tex=S):是**Acquisition Function(采集函数)**，这个函数的作用是用来选择公式(1)中的![[公式]](https://www.zhihu.com/equation?tex=x)，后面会详细介绍这个函数。

![[公式]](https://www.zhihu.com/equation?tex=%5Ccal%7BM%7D):是对数据集![[公式]](https://www.zhihu.com/equation?tex=D)进行拟合得到的模型。

重复这些步骤，直到达到某种收敛准则。我们用一句话来概括它，即如下所示

#### 在有限的迭代轮数内，按照损失函数的期望值最小同时方差最大的方式选择参数。直观点理解，就是选择loss小，并且最有可能更小的地方进行探索，寻找更优超参。

#### 2.1TPE在NNI中的使用

要使用 TPE，需要在 Experiment 的 YAML 配置文件进行如下改动：

```shell
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
    parallel_optimize: True
    constant_liar_type: min
```

**classArgs 要求：**

- **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 会试着最大化指标。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
- **parallel_optimize** (*bool, 可选, 默认值为 False*) - 如果为 True，TPE 会使用 Constant Liar 算法来优化并行超参调优。 否则，TPE 不会区分序列或并发的情况。
- **constant_liar_type** (*min、max 或 mean, 可选, 默认值为 min*) - 使用的 constant liar 类型，会在 X 点根据 y 的取值来确定。对应三个值：min{Y}, max{Y}, 和 mean{Y}。

#### 2.2其他的优化算法

除了TPE以外，NNI还提供了包括Anneal，SMAC等一系列常用的自动调参的算法

具体请参考[NNI官方文档](https://nni.readthedocs.io/zh/latest/Tuner/HyperoptTuner.html)



## 3.内置Assessor

为了节省计算资源，NNI 支持提前终止策略，并且通过叫做 **Assessor** 的接口来执行此操作。

Assessor 从 Trial 中接收中间结果，并通过指定的算法决定此 Trial 是否应该终止。 一旦 Trial 满足了提前终止策略（这表示 Assessor 认为最终结果不会太好），Assessor 会终止此 Trial，并将其状态标志为 EARLY_STOPPED。

以进行CIFAR10的训练为例，我们的config.yml文件应当参照以下设置assessor

```shell
authorName: default
experimentName: example_pytorch_cifar10
trialConcurrency: 4
maxExecDuration: 100h
maxTrialNum: 8
#choice: local, remote, pai
trainingServicePlatform: local
searchSpacePath: my_search_space.json
#choice: true, false
useAnnotation: false



tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
assessor:
  #choice: Medianstop, Curvefitting
  builtinAssessorName: Curvefitting
  classArgs:
    epoch_num: 20
    threshold: 0.9
trial:
  command: python3 main.py
  codeDir: .
  gpuNum: 1
localConfig:
  maxTrialNumPerGpu: 2



```

## 4.神经网络架构搜索

#### 4.1概述

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012), [ENAS](https://arxiv.org/abs/1802.03268), [DARTS](https://arxiv.org/abs/1806.09055), [Network Morphism](https://arxiv.org/abs/1806.10282)和 [Evolution](https://arxiv.org/abs/1703.01041)。 此外，新的创新不断涌现。

但是，要实现 NAS 算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。 为了促进 NAS 创新（例如，设计、实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

以此为动力，NNI 的目标是提供统一的体系结构，以加速 NAS 上的创新，并将最新的算法更快地应用于现实世界中的问题上。

通过统一的接口，有两种方法来使用神经网络架构搜索。 一种称为 [one-shot NAS](https://nni.readthedocs.io/zh/latest/NAS/Overview.html#supported-one-shot-nas-algorithms) ，基于搜索空间构建了一个超级网络，并使用 one-shot 训练来生成性能良好的子模型。 [第二种](https://nni.readthedocs.io/zh/latest/NAS/Overview.html#supported-classic-nas-algorithms) 是经典的搜索方法，搜索空间中每个子模型作为独立的 Trial 运行。 称之为经典的 NAS。

我们下面也着重介绍经典的NAS在NNI中的使用

#### 4.2神经网络架构搜索在NNI中的使用

首先编写搜索空间。

通常，搜索空间是要在其中找到最好结构的候选项。 无论是经典 NAS 还是 One-Shot NAS，不同的搜索算法都需要搜索空间。 NNI 提供了统一的 API 来表达神经网络架构的搜索空间。


```shell
  "lr": {
    "_type": "choice",
    "_value": [ 0.1, 0.0001 ]
  },
  "optimizer": {
    "_type": "choice",
    "_value": [ "SGD", "Adadelta", "Adagrad", "Adam", "Adamax" ]
  },
  "epochs": {
    "_type": "choice",
    "_value": [ 10 ]
  },
  "model": {
    "_type": "choice",
    "_value": [ "resnet18" ]
  },
  "dropout_rate": {
    "_type": "uniform",
    "_value": [ 0.1, 0.5 ]
  },
  "conv_size": {
    "_type": "choice",
    "_value": [ 2, 3, 5, 7 ]
  },
  "hidden_size": {
    "_type": "choice",
    "_value": [ 124, 512, 1024 ]
  },
  "batch_size": {
    "_type": "choice",
    "_value": [ 128,64 ]
  }
```

之后在我们的main.py函数中会依次使用model中的网络结构，并择优选择。详情请见[code](https://github.com/xmx-521/NNI-Student-Program-2020/tree/dev/Task2/Task2.2)

## 5.应用NNi于CIFAR10

​	实验环境:

- System:Ubuntu
- NNI version: 2.1
- Python version: 3.8.3
- Pytorch version: 1.6.0
- GPU Used:Yes
- Reverse Proxy Used:Yes
- Reverse Proxy Platform:frp



我们针对Resnet-18网络结构进行超参调优以及神经网络架构搜索，以找出学习率较高的Resnet-18网络结构参数。相应的搜索参数见上图的Resnet-18搜索空间部分。





![img](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20210314144605.png)



我们对以下八组参数组合进行实验，在12个EPOCH下观察Resnet-18对CIFAR10数据集的学习程度:

| Trial ID |   lr   | optimizer |    dropout_rate     | conv_size | hidden_size | batch_size |
| :------: | :----: | :-------: | :-----------------: | :-------: | :---------: | :--------: |
|    0     | 0.0001 |  Adagrad  | 0.16612533760388698 |     5     |    1024     |     64     |
|    1     | 0.0001 |  Adagrad  |  0.417230697474666  |     5     |    1024     |     64     |
|    2     |  0.1   |    SGD    | 0.43160786975356913 |     7     |     512     |     64     |
|    3     | 0.0001 |    SGD    |  0.438562283572701  |     2     |     124     |    128     |
|    4     | 0.0001 |   Adam    | 0.47975208536906955 |     2     |     124     |    128     |
|    5     |  0.1   |    SGD    | 0.45545744775165065 |     7     |     512     |     64     |
|    6     |  0.1   |    SGD    | 0.4890075286430945  |     2     |    1024     |     64     |
|    7     | 0.0001 |    SGD    | 0.19531881892955655 |     7     |     124     |    128     |

经过12轮EPOCH的训练后，几组Trial的学习效果如下：

| Trial ID | Metric |
| :------: | :----: |
|    0     | 56.48  |
|    1     | 59.67  |
|    2     | 75.56  |
|    3     | 64.10  |
|    4     | 86.12  |
|    5     | 75.36  |
|    6     | 77.83  |
|    7     | 63.41  |

NNI Immediate Metric图线展示：

![img](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20210314151756.png)





## 6.实验总结与心得

#### 6.1 在实验过程中存在的一些问题

- 由于设备机能以及时间有限，我们只能对Resnet18这样相对浅层的网络进行样本量较少的HPO与NAS调试，且训练的EPOCH相对较少，因此目前所得到的结果（见上Trial ID 4）从理论上对于Resnet18对CIFAR10的训练仍有较大的提升空间.

- 在NNI Web 界面中出现的一些问题：我们使用第三方平台（易学智能）的付费GPU进行实验，并通过部署frp反向代理软件将训练结果呈现在NNI Web上，通过此方式创建的NNI Web界面的Hyper-parameter栏图表无法正常呈现。NNI文档上对使用frp访问Web界面的引导较为简略，我们也希望官方未来能添加相关内容，为NNI在frp的部署上提供便利。

  

#### 6.2 实验的心得体会

​	   通过本次实验，我们感受到了NNI作为AutoML工具轻量级、简洁性的优势。自动模型调优功能的部署，可以省下较多的人工调参的经历，因此未来可以积极地部署在其他方面（如学校实验室科研）的领域。我们在Task 4中也将利用NNI这一工具，逐步完善我们对自选课题的模型准确率的提升。



