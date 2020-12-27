# Task 2.1 **进阶任务项目说明文档

——基于自动机器学习工具 NNI 的创新性科研扩展项目说明

---

## 团队基本信息

- 团队名：电脑一带五
- 团队成员：徐满心、李乐天、孟繁青、尚丙奇、陈垲昕
- 团队学校：同济大学
- Github Repo地址: https://github.com/xmx-521/NNI-Student-Program-2020

## 任务要求：NNI-task2.1 进阶任务

### 文档情况

- [X] CIFAR10图像分类测试文档：
  包括工具比较、安装使用等。

### 文档内容描述

本文档为CIFAR10图像分类样例分析报告，具体内容包括

- CIFAR10简介
- 加载和初始化标准化CIFAR10训练和测试数据集 torchvision。
- 定义卷积神经网络
- 定义损失函数
- 根据训练数据训练网络
- 在测试数据上测试网络
- 在GPU上进行上述操作

## 1.CIFAR10简介

CIFAR10，该数据集共有60000张彩色图像，这些图像是32*32*3（记住这个32*32很重要），分为10个类，每类6000张图。这里面有50000张用于训练，构成了5个训练批，每一批10000张图；另外10000用于测试，单独构成一批。测试批的数据里，取自10类中的每一类，每一类随机取1000张。抽剩下的就随机排列组成了训练批。注意一个训练批中的各类图像并不一定数量相同，总的来看训练批，每一类都有5000张图。

下面是一张广为流传的CIFAR10的图像

[![rUmlr9.png](https://s3.ax1x.com/2020/12/19/rUmlr9.png)](https://imgchr.com/i/rUmlr9)

## 2.加载和初始化CIFAR10训练和测试数据集torchvision

```shell
import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

#### 2.1代码分析

我们通过torchvision来装载数据集，使得加载CIFAR10非常容易。
另外需要注意的是由于torchvision输出的数据集是[0,1]的PILImage图像，我们将其转化为归一化繁为[-1,1]的tensor。

#### 2.2代码结果展示

```shell
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

[![rUmsat.md.png](https://s3.ax1x.com/2020/12/19/rUmsat.md.png)](https://imgchr.com/i/rUmsat)

```shell
output: car  deer  frog   car
```

## 3.定义并初始化卷积神经网络

```shell
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(3, 6, 5)
        ### 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即RGB图）,输出为 6张特征图, 卷积核为5x5正方形
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2 = nn.Linear(120, 84)
        # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3 = nn.Linear(84, 10)
        # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。

#定义神经网络的向前传播函数，一旦成功定义，其向后传播函数也会自动生成。
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

```

#### 3.1代码分析

首先我们要熟悉Conv2d,MaxPool2d的用法

```shell
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

```

第一次卷积：输入的channel为3，输出的1channel为6，使用6个大小为5 x 5的卷积核，故卷积核的规模为(5 x 5) x 6；卷积操作的stride默认值为1 x 1，32 - 5 + 1 = 28，并且使用ReLU对第一次卷积后的结果进行非线性处理，输出大小为28 x 28 x 6。

第一次卷积后最大池化：kernel_size为2 x 2，输出大小变为14 x 14 x 6。

第二次卷积：输入是14 x 14 x 6，又因为输出的channel为16，使用了16个卷积核，计算（14 - 5）/ 1 + 1 = 10，那么通过conv2输出的结果是10 x 10 x 16。

第二次卷积后最大池化：输入是10 x 10 x 16，窗口2 x 2，计算10 /  2 = 5，那么通过max_pool2层输出结果是5 x 5 x 16。

第一次全连接：将上一步得到的结果铺平成一维向量形式，5 x 5 x 16 = 400，即输入大小为400 x 1，W大小为120 x 400，输出大小为120 x 1；

第二次全连接：W大小为84 x 120，输入大小为120 x 1，输出大小为84 x 1；

第三次全连接：W大小为10 x 84，输入大小为84 x 1，输出大小为10 x 1，即分别预测为10类的概率值。

#### 3.2卷积网络示意图

[![rUmN8O.png](https://s3.ax1x.com/2020/12/19/rUmN8O.png)](https://imgchr.com/i/rUmN8O)

## 4.定义损失函数与优化器

```shell
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

## 5.训练网络

#### 5.1训练代码

```shell
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

#### 5.2训练结果

[![rUm4qs.jpg](https://s3.ax1x.com/2020/12/19/rUm4qs.jpg)](https://imgchr.com/i/rUm4qs)

## 6.测试效果

我们已经在训练数据集中对网络进行了2次训练。但是我们需要检查网络是否学到了什么。
我们将通过预测神经网络输出的类标签并根据实际情况进行检查来进行检查。如果预测正确，则将样本添加到正确预测列表中。
好的，第一步。让我们显示测试集中的图像以使其熟悉，同时保存我们的模型。

#### 6.1测试代码

```shell
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
net = Net()
net.load_state_dict(torch.load(PATH))
```

之后看看该网络在整个数据集上的表现

```shell
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

最后看看该网络在哪些类上具有较高正确率，而哪些类较低

```shell
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
  

```

#### 6.2训练结果

[![rUmXM4.jpg](https://s3.ax1x.com/2020/12/19/rUmXM4.jpg)](https://imgchr.com/i/rUmXM4)

## 7.GPU版本

在进行此步骤时，我们遇到了一个问题

```shell
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
```

最后经过我们查阅相关资料，发现需要加上下述语句，才可正常运行

```shell
if __name__ == '__main__':
```

#### 7.1采用GPU之前的代码

代码实际上就是上述各部分的组合，由于过长，上传至code文件夹。

#### 7.2采用GPU之前的代码运行结果

[![rUmiuj.md.png](https://s3.ax1x.com/2020/12/19/rUmiuj.md.png)](https://imgchr.com/i/rUmiuj)

#### 7.2采用GPU之后的代码

详情请见code文件夹中CIFAR10_GPU_After.py

#### 7.4采用GPU后的代码运行结果

[![rUmu2F.md.png](https://s3.ax1x.com/2020/12/19/rUmu2F.md.png)](https://imgchr.com/i/rUmu2F)

#### 7.5实验结果分析

我们在GPU版本中尝试增加网络的宽度（第一个参数2`nn.Conv2d`和第二个参数1 `nn.Conv2d`–它们必须是相同的数字），看看会得到什么样的加速。

但是我们发现，有无GPU的版本在运行时间上几乎相差无几，GPU版本仅仅是在准确率上稍微优于无GPU版本。

可能的原因是：我们发现运行过程中，由于该网络过于简单，且GPU的内存占有率并不高，导致这两个版本并没有体现出明显的差异。
