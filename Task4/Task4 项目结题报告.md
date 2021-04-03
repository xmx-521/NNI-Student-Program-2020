# Task4 项目结题报告

## 团队基本信息

- 团队名：电脑一带五
- 团队成员：徐满心、李乐天、孟繁青、尚丙奇、陈垲昕
- 团队学校：同济大学
- Github Repo地址: https://github.com/xmx-521/NNI-Student-Program-2020

## 任务要求：NNI-Task 4 自主任务

将 NNI 学生项目与学校或实验室项目进行互补结合，进行 NNI 的实践性操作与拓展性应用。





## 代码实现

### 原始网络实现



<img src="https://pic.downk.cc/item/5fddcda73ffa7d37b3888504.png">

我们已经在Task4的项目开题报告中阐述了我们初步构建的深度神经网络结构。具体到实现上，我们需要定义两个从 nn.Module 中继承的类：1.用于特征提取，解决特征消失问题，增加网络深度的残差块 Resblock ，2.我们的深度神经网络整体架构，引入 Resblock 作为网络结构组成部分。

残差块的定义见下方代码，我们实现了残差块的初始化（卷积核大小为 [![img](https://camo.githubusercontent.com/3f087594c1cd30b5b2c58e68b9e90e79a46d95cbecd6211fd50b9381efcc27f0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b332a33)](https://camo.githubusercontent.com/3f087594c1cd30b5b2c58e68b9e90e79a46d95cbecd6211fd50b9381efcc27f0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b332a33) ，通道数为 [![img](https://camo.githubusercontent.com/7322df940cd807eeb75bc6be951980135161bc3ec601c47a88ea0f2c16973ace/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b353132)](https://camo.githubusercontent.com/7322df940cd807eeb75bc6be951980135161bc3ec601c47a88ea0f2c16973ace/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b353132) ）以及对应的前向传播方法:

```python
class Resblock(nn.Module):

    def __init__(self):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        # 8*8*512,kernel=3,strides=1,计算padding=1

    def forward(self, x):
        out = x
        out = F.relu(self.conv1(out))
        out = F.relu(F.relu(self.conv2(out))+x)

        return out
```

而在原始网络中，设置了卷积核大小为 [![img](https://camo.githubusercontent.com/d1dbba0901789999a9bc7d3ff48b8271e071c322bce68a98710fc16fcc034b45/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b372a37)](https://camo.githubusercontent.com/d1dbba0901789999a9bc7d3ff48b8271e071c322bce68a98710fc16fcc034b45/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b372a37)、[![img](https://camo.githubusercontent.com/3f087594c1cd30b5b2c58e68b9e90e79a46d95cbecd6211fd50b9381efcc27f0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b332a33)](https://camo.githubusercontent.com/3f087594c1cd30b5b2c58e68b9e90e79a46d95cbecd6211fd50b9381efcc27f0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b332a33)、[![img](https://camo.githubusercontent.com/3f087594c1cd30b5b2c58e68b9e90e79a46d95cbecd6211fd50b9381efcc27f0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b332a33)](https://camo.githubusercontent.com/3f087594c1cd30b5b2c58e68b9e90e79a46d95cbecd6211fd50b9381efcc27f0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b332a33)、[![img](https://camo.githubusercontent.com/3f087594c1cd30b5b2c58e68b9e90e79a46d95cbecd6211fd50b9381efcc27f0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b332a33)](https://camo.githubusercontent.com/3f087594c1cd30b5b2c58e68b9e90e79a46d95cbecd6211fd50b9381efcc27f0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b332a33) 的四层卷积层，其通道数分别为 [![img](https://camo.githubusercontent.com/744a049a90a2184381826f2627c63be08fe21d5c04daf7729d2e09247001d8f0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b3634)](https://camo.githubusercontent.com/744a049a90a2184381826f2627c63be08fe21d5c04daf7729d2e09247001d8f0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b3634)、[![img](https://camo.githubusercontent.com/7d654a3f968a292f1306c80588ed6ab37941731a355903a5dbe1810c94537d99/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b313238)](https://camo.githubusercontent.com/7d654a3f968a292f1306c80588ed6ab37941731a355903a5dbe1810c94537d99/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b313238)、[![img](https://camo.githubusercontent.com/e00775ac8c0cee7bb0c694ef833d6b527c1a995ce1c95f206f10ad8b309d399c/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b323536)](https://camo.githubusercontent.com/e00775ac8c0cee7bb0c694ef833d6b527c1a995ce1c95f206f10ad8b309d399c/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b323536) 和 [![img](https://camo.githubusercontent.com/7322df940cd807eeb75bc6be951980135161bc3ec601c47a88ea0f2c16973ace/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b353132)](https://camo.githubusercontent.com/7322df940cd807eeb75bc6be951980135161bc3ec601c47a88ea0f2c16973ace/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b353132)。接着设置两个我们先前定义的 ResBlock 层。接下来两个全连接层的大小为 [![img](https://camo.githubusercontent.com/2907fc40036c5fae5bdb88dade4bc04914419a55cd42c743069e1070ebcf8b60/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b31303234)](https://camo.githubusercontent.com/2907fc40036c5fae5bdb88dade4bc04914419a55cd42c743069e1070ebcf8b60/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b31303234) 和 [![img](https://camo.githubusercontent.com/7322df940cd807eeb75bc6be951980135161bc3ec601c47a88ea0f2c16973ace/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b353132)](https://camo.githubusercontent.com/7322df940cd807eeb75bc6be951980135161bc3ec601c47a88ea0f2c16973ace/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f2535436c617267652673706163653b353132) 个节点，最后经过 softmax 函数，输出识别结果。图像依次通过上述结构，最终得到识别。具体见下方代码：

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(4, 64, 7, 2, 3)
        # 定义conv1函数的是图像卷积函数：输入2个channel,输出为64个channel
        # 卷积核为7x7正方形,strides=2,padding经计算为3
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        # 定义conv2函数的是图像卷积函数：输入64个channel,输出为128个channel
        # 卷积核为3x3正方形,strides=2,paddomh=1
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        # 定义conv3函数的是图像卷积函数：输入128个channel,输出为256个channel
        # 卷积核为5x5正方形,strides=2,padding=1
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        # 同上
        self.resblock1 = Resblock()
        self.resblock2 = Resblock()
        self.fc1 = nn.Linear(8*8*512, 1024)
        # 全联接层
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 21)

        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # 先卷积在激活
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # 到残差块
        x = self.resblock1(x)
        x = self.resblock2(x)
        # 做两次，传出激活后的值
        x = x.view(-1, 8*8*512)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

而在数据集处理方面，我们将原数据集通过 preprocess.py 转存为 hdf5 格式文件存储（详见本项目code/utils中的源码，此处不再赘述 )，并实现 EVimageDataset 数据集，通过传入 hdf5 数据集文件路径进行初始化。

我们的数据集一共有21个测试者的步态信息，每个测试者的步态信息有100个关键帧，EVimageDataset 存储的是所有的关键帧，其中以100个关键帧为一组 ( 即：0-99,100-199,....,2000-2099分别为每个测试者的步态信息 ) 。在调用 getitem方法时，我们先通过数据集的索引定位测试者序号 (person_index)，再获取对应索引的图像信息与标签信息。

```python
class EVimageDataset(Dataset):

    def __init__(self, path):
        # input:the path of hdf5 file
        self.path = path

    def __len__(self):
        return 2100

    def __getitem__(self, i):
        # input an index of 0-2099,representing
        person_index = (int)(i/100)

        # get person label tensor
        label = torch.tensor(person_index, dtype=torch.int64)

        # open hdf5 file
        f1 = h5py.File(self.path, 'r')

        # get group index
        grp = f1[str(person_index)]

        # get dataset with image_index
        image_index = i % 100+1
        dset = grp[str(image_index)]

        # get dataset tensor
        image = dset[()]

        return image, label
```

在网络训练函数 train_net中 (函数位于model.py)，我们使用torch.optim中的Adam优化器，以及分类交叉熵损失函数：

```python
optimizer = torch.optim.Adam(net.parameters(), lr=lr,  weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'max', patience=2)
criterion = nn.CrossEntropyLoss()
```

最后，我们在主函数中完成训练

```python
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:1'if torch.cuda.is_available()else 'cpu')
    logging.info(f'Using device {device}')

    net = Net()
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
```



### NNI模块嵌入

我们在Task4项目开题报告中已经汇报了我们已有的成果：在不使用去噪算法（见原论文）的情况下达到了85%左右的识别准确率。我们希望能通过NNI，通过对网络的调整进一步提升识别准确率。

在主网络定义完成后，我们引入NNI模块，使用NNI的HPO超参调优功能，对该网络的设置以及训练信息进行调整

在config.yml中，我们设置训练时使用本地的GPU，以提升训练速度：

```yaml
localConfig:
   useActiveGpu: true
   maxTrialPerGpu: 1
```

在搜索空间中，我们定义了如下可选的调整参数：

```json
{
    "lr":            { "_type": "uniform", "_value": [ 3e-7,3e-2 ] },
    "batch_size":    { "_type": "randint", "_value": [ 16  ,256  ] },
    "dropout_rate":  { "_type": "uniform", "_value": [ 0   ,0.99 ] },
    "hidden_size":   { "_type": "randint", "_value": [ 512 ,2048 ] }
}
```

我们在主函数中，使用nni的get_next_parameter()获取一组超参组合，结果存在nni_args中，这个参数组合将用于网络的初始化以及网络训练函数的参数设置：

```python
nni_args = nni.get_next_parameter()
```

接着，我们将参数引入网络的初始化中。在model.py中，网络的初始化使用了传入的hidden_size与dropout_rate

```python
#重写fc1,fc2，引入nni_args['hidden_size']
self.fc1 = nn.Linear(8*8*512, hidden_size)
self.fc2 = nn.Linear(hidden_size, 512)
        
#重写dropout1,引入nni_args['dropout_rate']
self.dropout1 = nn.Dropout(dropout_rate)
```

```python
net = Net(dropout_rate=nni_args["dropout_rate"],hidden_size=nni_args["hidden_size"])
```

而在网络训练函数中，引入搜索空间参数组合的batch_size与lr:

```python
train_net(net=net,
          epochs=args.epochs,
          batch_size=nni_args["batch_size"],
          lr=nni_args["lr"],
          device=device)
```

并使用nni模块，在每组实验完成后回馈训练的最佳结果：

```python
 nni.report_final_result(best_acc)
```

### NNI运行

#### 实验环境

NNI version: 2.1

NNI mode: local

Client OS: Ubuntu 18.04

Python version: 3.7.8

PyTorch version: 1.8.1+cuda11

Is conda /virtualenv /env used?: Yes

Is running in Docker?: No

Is GPU used?:Yes,GeForce RTX 3090*1



config.yml配置：

```yaml
authorName: default
experimentName: EVImageCamera_HPO
trialConcurrency: 1
maxExecDuration: 10h
maxTrialNum: 2000
#choice: local, remote
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
  command: python3 train.py
  codeDir: .
  gpuNum: 1

localConfig:
  useActiveGpu: true
  maxTrialPerGpu: 1
```

#### 第一次实验

第一次实验时，我们选择了较大的搜索空间，防止NNI卡在局部最优解，搜索空间如下：

```json
{
   "lr": {
       "_type": "uniform",
       "_value": [
           3e-7,
           3e-2
       ]
   },
   "batch_size": {
       "_type": "randint",
       "_value": [
           16,
           256
       ]
   },
   "dropout_rate": {
       "_type": "uniform",
       "_value": [
           0,
           0.99
       ]
   },
   "hidden_size": {
       "_type": "randint",
       "_value": [
           512,
           2048
       ]
   }
}
```

实验结果如下：

[![cnKMs1.png](https://z3.ax1x.com/2021/04/03/cnKMs1.png)](https://imgtu.com/i/cnKMs1)

本次实验最高准确率约为88.3%，下图蓝色曲线为原论文中准确率，可以看出经过NNI初步调参后，gait准确率几乎达到甚至略微超越了原论文的水准，然而优势并没有那么明显，因此我们根据以上的参数进行搜索空间的缩小，进行第二次超参调优

[![cn3tKJ.png](https://z3.ax1x.com/2021/04/03/cn3tKJ.png)](https://imgtu.com/i/cn3tKJ)

#### 第二次实验

我们根据第一次实验中准确率Top 5%的trials的参数，缩小搜索空间，试图找到更优的解，搜索空间如下：

```json
{
    "lr": {
        "_type": "uniform",
        "_value": [
            3e-3,
            4.5e-3
        ]
    },
    "batch_size": {
        "_type": "randint",
        "_value": [
            115,
            125
        ]
    },
    "dropout_rate": {
        "_type": "uniform",
        "_value": [
            0.10,
            0.20
        ]
    },
    "hidden_size": {
        "_type": "randint",
        "_value": [
            750,
            1000
        ]
    }
}
```

最终，我们得到了高达89.8%的准确率，明显高于原论文最高88%-89%左右的准确率。

[![cn3qqs.png](https://z3.ax1x.com/2021/04/03/cn3qqs.png)](https://imgtu.com/i/cn3qqs)

[![cn3zGT.png](https://z3.ax1x.com/2021/04/03/cn3zGT.png)](https://imgtu.com/i/cn3zGT)



更多实验结果如下：

[![cn8pzF.png](https://z3.ax1x.com/2021/04/03/cn8pzF.png)](https://imgtu.com/i/cn8pzF)

[![cn8PsJ.png](https://z3.ax1x.com/2021/04/03/cn8PsJ.png)](https://imgtu.com/i/cn8PsJ)

[![cn8Zi6.png](https://z3.ax1x.com/2021/04/03/cn8Zi6.png)](https://imgtu.com/i/cn8Zi6)

[![cn8eJK.png](https://z3.ax1x.com/2021/04/03/cn8eJK.png)](https://imgtu.com/i/cn8eJK)

[![cn8mRO.png](https://z3.ax1x.com/2021/04/03/cn8mRO.png)](https://imgtu.com/i/cn8mRO)
