# Task 1 **入门任务** 使用体验文档

——基于自动机器学习工具 NNI 的创新性科研扩展项目说明

------







## 团队基本信息

- 团队名：电脑一带五

- 团队成员：徐满心、李乐天、孟繁青、尚丙奇、陈垲昕

- 团队学校：同济大学

- Github Repo地址: https://github.com/xmx-521/NNI-Student-Program-2020

   



## 1.任务要求：NNI-task1 入门任务

### 文档情况

- [x] NNI使用体验文档：
  包括工具比较、安装使用等。
- [ ] NNI 样例分析文档：
  包括运行代码、实验结果、样例分析文档的md文件。

### 文档内容描述

​	本文档为NNI使用体验文档报告，具体内容包括

- AutoML工具比较：调研流行的 AutoML框架，分析 NNI 的优势和劣势。
- NNI安装与使用（平台基本信息，安装流程，运行问题以及解决方案）
- NNI使用感受

## 2.AutoML 工具比较

### 1.TPOT

![14](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211201.webp)

#### 简介

​	TPOT是一个使用genetic programming算法优化机器学习piplines的Python自动机器学习工具，通过智能地探索数千种可能的piplines来为数据找到最好的一个，从而自动化机器学习中最乏味的部分。当TPOT完成搜索的时候，它会为用户提供Python代码，以便找到最佳的管道，这样用户就可以从那里修补管道。

#### 输出结果

最佳模型组合及其参数(python文件)和最佳得分。

#### 优劣

tpot在数据治理阶段采用了PCA主成份分析，在模型选择过程中可以使用组合方法，分析的过程比起其他工具更科学，并能直接生成一个写好参数的python文件，但输出可参考的结果较少，不利于进一步分析。



### 2.Auto_Sklearn

![15](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211202.jpg)

#### 简介

​		auto-sklearn将机器学习用户从算法选择和超参数调整中解放出来。它利用了最近在贝叶斯优化、元学习和集成构建方面的优势，主要使用穷举法在有限的时间内逐个尝试最优模型。

#### 输出结果

计算过程以及最终模型的准确率。

#### 优劣

穷举法在时间充裕的情况下可以加大预算周期不断让机器尝试最优解，但输出结果较少。

### 3.Advisor

![16](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211203.jpg)

#### 简介：

Advisor是用于黑盒优化的调参系统。它是Google Vizier的开源实现，编程接口与Google Vizier相同。

#### 输出结果：

推荐参数与训练模型。

#### 优劣：

方便与API、SDK、WEB和CLI一起使用，支持研究和试验抽象化，包括搜索和early stop算法，像Microsoft NNI一样的命令行工具。



## 3.NNI与安装使用

### 3.1 基本信息

- NNI version: **1.8**
- conda version: **4.9.2**
- NNI mode (local|remote|pai): **Local**
- Client OS: **Windows**
- Python version: **3.7.9**
- PyTorch version: **1.7.0**
- Is conda/virtualenv/venv used?: **yes**
- Is running in Docker?: **no**

### 3.2 安装NNI流程

![flow](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211341.png)



#### 3.2.1创建虚拟环境

```shell
conda create -n nnitorch python=3.7
conda activate nnitorch
```

#### 3.2.2安装nni与torch环境		

```Shell
pip install nni==1.8 torch==1.7.0
```

#### 3.2.3安装样例并运行	

```shell
git clone -b v1.8 https://github.com/microsoft/nni.git E:\nniGit
nnictl create --config E:\nniGit\examples\trials\mnist-pytorch\config_windows.yml
```

​	

### 3.3 安装流程出错以及解决方案

​		本小组在安装nni环境并运行样例程序的过程中，曾遇到了一些问题导致nni装载失败或样例运行失败，现将安装过程中出现过的问题以及解决方案予以记录

#### All Trials Failed	

#### 出错实验配置文件

- **..\mnist-pytorch\config_windows.yml**

```yaml
authorName: default
experimentName: example_mnist_pytorch
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: local
searchSpacePath: search_space.json

#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python mnist.py
  codeDir: .
  gpuNum: 0
```

#### 问题简述

​	在安装nni环境与pytorch环境后，nnictl运行mnist-pytorch\config_windows.yml

```
git clone -b v1.8 https://github.com/microsoft/nni.git E:\nniGit
nnictl create --config E:\nniGit\examples\trials\mnist-pytorch\config_windows.yml
```

​	样例成功运行(**Status:DONE**)但所有Trials都未成功运行:每组样例的状态从WAITING切换至RUNNING，在数秒后切换为FAILED

#### 错误图例与报告

- ​	Overview:

![1](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211204.png)

- ​	Trials jobs info:

![2](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211205.png)

注：Dispatcher Log 与 NNI Manager Log 见 experiment info\error experiment info 文件夹





#### 一种可能问题的分析与解决方案	

- 原因：nni_experiements运行环境路径无法解析（可能是中文路径导致的乱码）

查看nni_experiments下当前实验的错误运行报告stderr，发现实验并没有成功记录stderr，在windows powershell下运行当前实验目录下的run.ps1发现无法执行

- 解决方案

在配置文件**..\mnist-pytorch\config_windows.yml **中加入以下代码：

```yaml
...

#reset nni_experiment dir
logDir: E:/nniGit_experiments
logLevel: info

...
```

转换nni_experiment本地实验信息记录路径后，再次执行:

```shell
nnictl create --config E:\nniGit\examples\trials\mnist-pytorch\config_windows.yml
```

成功生成实验，Trials正常RUNNING并最终SUCCESS

![5](https://cdn.jsdelivr.net/gh/MountPOTATO/pic/img/20201227211206.png)





### 3.4 使用心得

#### 3.4.1 对NNI自身优点的一些心得体会

通过对NNI的初步使用，我们可以切身体会到NNI的一些使用优势

- 安装简洁：作为轻量级AutoML工具，NNI的安装简洁而容易，且只需一行命令即可提交训练；
- 支持私有部署，用自己的计算资源就能进行自动机器学习；
- 支持分布式调度：NNI 可以在单机上进行试验，还支持多种分布式调度平台，如通过 SSH 控制多台 GPU 服务器协同完成试验，或通过OpenPAI, Azure, Kubernetes等在独立的 Docker 中运行；
- 对超参搜索的底层支持：大部分自动机器学习服务与工具的服务都是针对于图片分类等特定的任务。而NNI通过让用户提供训练代码和超参搜索范围， 依靠NNI内置算法和训练服务来搜索最佳的超参和神经架构，NNI为有一定模型训练经验的用户提供更加通用的调参方式，便于用户做出更多尝试、决策和加入思考，并完整参与整个训练过程；
- 随库自带众多实例和流行的调参算法；
- 架构简洁、可视化界面明晰易懂，对开发和扩展及其友好。
- github的issue问题整理与管理较好，对用户在使用过程中出现的问题有细致而及时的回答

#### 3.4.2 对NNI改进的一点意见

小组在对NNI使用的基础上，对NNI提出一些改进的意见

- 完善安装问题分析文档：目前的安装问题分析与解决方案仍然需要进一步更新并提出更细致的解答
- Web UI端功能的多元化：在调参数据的可视化上可以进行进一步的优化