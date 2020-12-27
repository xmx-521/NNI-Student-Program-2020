# Task 1 **入门任务** 样例分析文档

——基于自动机器学习工具 NNI 的创新性科研扩展项目说明

------

## 团队基本信息

- 团队名：电脑一带五
- 团队成员：徐满心、李乐天、孟繁青、尚丙奇、陈垲昕
- 团队学校：同济大学
- Github Repo地址: https://github.com/xmx-521/NNI-Student-Program-2020

   



## 1.任务要求：NNI-task1 入门任务

### 文档情况

- [ ] NNI使用体验文档：
  包括工具比较、安装使用等。
- [x] NNI 样例分析文档：
  包括运行代码、实验结果、样例分析文档的md文件。

### 文档内容描述





## 2.mnist-pytorch样例测试流程

### 2.1 基本信息

- 测试平台：	 **Windows 10**
- 测试环境：     **Anaconda 3** 
- conda版本：  **4.9.2**
- nni版本：       **1.8**
- torch版本：   **1.7.0**

### 2.2 基本运行代码

#### 2.2.1 配置文件：mnist-pytorch\config_windows.yml

```yaml
authorName: default
experimentName: example_mnist_pytorch
trialConcurrency: 1						#同时运行的Trial任务的最大数量为1。
maxExecDuration: 2h						#整个调参过程最长运行时间为2h
maxTrialNum: 10							#NNI创建的最大Trial任务数
#choice: local, remote, pai
trainingServicePlatform: local			#指定local为运行Experiment 的平台
searchSpacePath: search_space.json		#搜索空间文件

#reset nni_experiment dir
logDir: E:/nniGit_experiments			#实验日志路径
logLevel: info

#choice: true, false
useAnnotation: false					#不删除searchSpacePath字段
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
  #SMAC (SMAC should be installed through nnictl)
  
  builtinTunerName: TPE					#指定内置的调参算法为TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python mnist.py
  codeDir: .							#指定Trial文件的目录
  gpuNum: 0								#运行每个Trial进程的GPU数量为0
```

#### 2.2.2 搜索空间：search_space.json

```yaml
{
	#定义一次训练所选取的样本数
    "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
    #定义隐藏层尺寸
    "hidden_size":{"_type":"choice","_value":[128, 256, 512, 1024]},
    #定义学习率
    "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
    #定义动量
    "momentum":{"_type":"uniform","_value":[0, 1]}
}
```

设置训练次数：10；最长运行时间：2h ；tuner: TPE。



### 2.3 实现与运行结果

​	Overview

![4](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211420.png)

   Trial Detail & Default Metric

![7](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211421.png)

![6](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211422.png)



Hyper Parameter:

![9](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211423.png)



Trial Duration:

![10](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211424.png)



Intermediate result

![11](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211425.png)