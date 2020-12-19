# Task 2.1 **进阶任务项目说明文档

——基于自动机器学习工具 NNI 的创新性科研扩展项目说明

------

[TOC]

## 团队基本信息

- 团队名：电脑一带五
- 团队成员：徐满心、李乐天、孟繁青、尚丙奇、陈垲昕
- 团队学校：同济大学
- Github Repo地址: https://github.com/xmx-521/NNI-Student-Program-2020







## 任务要求：NNI-task2.1 进阶任务

### 文档情况

- [x] CIFAR10图像分类测试文档：
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

```shell

if __name__ == '__main__':
    # -*- coding: utf-8 -*-
    """
    Training a Classifier
    =====================

    This is it. You have seen how to define neural networks, compute loss and make
    updates to the weights of the network.

    Now you might be thinking,

    What about data?
    ----------------

    Generally, when you have to deal with image, text, audio or video data,
    you can use standard python packages that load data into a numpy array.
    Then you can convert this array into a ``torch.*Tensor``.

    -  For images, packages such as Pillow, OpenCV are useful
    -  For audio, packages such as scipy and librosa
    -  For text, either raw Python or Cython based loading, or NLTK and
    SpaCy are useful

    Specifically for vision, we have created a package called
    ``torchvision``, that has data loaders for common datasets such as
    Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
    ``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

    This provides a huge convenience and avoids writing boilerplate code.

    For this tutorial, we will use the CIFAR10 dataset.
    It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
    ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
    size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

    .. figure:: /_static/img/cifar10.png
    :alt: cifar10

    cifar10


    Training an image classifier
    ----------------------------

    We will do the following steps in order:

    1. Load and normalizing the CIFAR10 training and test datasets using
    ``torchvision``
    2. Define a Convolutional Neural Network
    3. Define a loss function
    4. Train the network on the training data
    5. Test the network on the test data

    1. Loading and normalizing CIFAR10
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Using ``torchvision``, it’s extremely easy to load CIFAR10.
    """
    import torch
    import torchvision
    import torchvision.transforms as transforms

    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    # .. note::
    #     If running on Windows and you get a BrokenPipeError, try setting
    #     the num_worker of torch.utils.data.DataLoader() to 0.

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

    ########################################################################
    # Let us show some of the training images, for fun.

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


    ########################################################################
    # 2. Define a Convolutional Neural Network
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Copy the neural network from the Neural Networks section before and modify it to
    # take 3-channel images (instead of 1-channel images as it was defined).

    import torch.nn as nn
    import torch.nn.functional as F


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = Net()
    net.to(device=device)

    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs=inputs.to(device=device)
            labels=labels.to(device=device)
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

    ########################################################################
    # Let's quickly save our trained model:

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    ########################################################################
    # See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_
    # for more details on saving PyTorch models.
    #
    # 5. Test the network on the test data
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # We have trained the network for 2 passes over the training dataset.
    # But we need to check if the network has learnt anything at all.
    #
    # We will check this by predicting the class label that the neural network
    # outputs, and checking it against the ground-truth. If the prediction is
    # correct, we add the sample to the list of correct predictions.
    #
    # Okay, first step. Let us display an image from the test set to get familiar.

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    ########################################################################
    # Next, let's load back in our saved model (note: saving and re-loading the model
    # wasn't necessary here, we only did it to illustrate how to do so):

    net = Net()
    net.load_state_dict(torch.load(PATH))

    ########################################################################
    # Okay, now let us see what the neural network thinks these examples above are:

    outputs = net(images)

    ########################################################################
    # The outputs are energies for the 10 classes.
    # The higher the energy for a class, the more the network
    # thinks that the image is of the particular class.
    # So, let's get the index of the highest energy:
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))

    ########################################################################
    # The results seem pretty good.
    #
    # Let us look at how the network performs on the whole dataset.
    net.to(device=device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images=images.to(device=device)
            labels=labels.to(device=device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    ########################################################################
    # That looks way better than chance, which is 10% accuracy (randomly picking
    # a class out of 10 classes).
    # Seems like the network learnt something.
    #
    # Hmmm, what are the classes that performed well, and the classes that did
    # not perform well:
    net.to(device=device)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images=images.to(device=device)
            labels=labels.to(device=device)
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

    ########################################################################
    # Okay, so what next?
    #
    # How do we run these neural networks on the GPU?
    #
    # Training on GPU
    # ----------------
    # Just like how you transfer a Tensor onto the GPU, you transfer the neural
    # net onto the GPU.
    #
    # Let's first define our device as the first visible cuda device if we have
    # CUDA available:

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    ########################################################################
    # The rest of this section assumes that ``device`` is a CUDA device.
    #
    # Then these methods will recursively go over all modules and convert their
    # parameters and buffers to CUDA tensors:
    #
    # .. code:: python
    #
    #     net.to(device)
    #
    #
    # Remember that you will have to send the inputs and targets at every step
    # to the GPU too:
    #
    # .. code:: python
    #
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #
    # Why dont I notice MASSIVE speedup compared to CPU? Because your network
    # is really small.
    #
    # **Exercise:** Try increasing the width of your network (argument 2 of
    # the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
    # they need to be the same number), see what kind of speedup you get.
    #
    # **Goals achieved**:
    #
    # - Understanding PyTorch's Tensor library and neural networks at a high level.
    # - Train a small neural network to classify images
    #
    # Training on multiple GPUs
    # -------------------------
    # If you want to see even more MASSIVE speedup using all of your GPUs,
    # please check out :doc:`data_parallel_tutorial`.
    #
    # Where do I go next?
    # -------------------
    #
    # -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
    # -  `Train a state-of-the-art ResNet network on imagenet`_
    # -  `Train a face generator using Generative Adversarial Networks`_
    # -  `Train a word-level language model using Recurrent LSTM networks`_
    # -  `More examples`_
    # -  `More tutorials`_
    # -  `Discuss PyTorch on the Forums`_
    # -  `Chat with other users on Slack`_
    #
    # .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
    # .. _Train a face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
    # .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
    # .. _More examples: https://github.com/pytorch/examples
    # .. _More tutorials: https://github.com/pytorch/tutorials
    # .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
    # .. _Chat with other users on Slack: https://pytorch.slack.com/messages/beginner/

    # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
    del dataiter
    # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%

```


#### 7.2采用GPU之前的代码运行结果
[![rUmiuj.md.png](https://s3.ax1x.com/2020/12/19/rUmiuj.md.png)](https://imgchr.com/i/rUmiuj)





#### 7.2采用GPU之后的代码

```shell

# -*- coding: utf-8 -*-
"""
Training a Classifier
=====================

This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.

Now you might be thinking,

What about data?
----------------

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful

Specifically for vision, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


Training an image classifier
----------------------------

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

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

########################################################################
# Let us show some of the training images, for fun.

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


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = Net()
net.to(device=device)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.to(device=device)
        labels=labels.to(device=device)
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

########################################################################
# Let's quickly save our trained model:

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

########################################################################
# See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_
# for more details on saving PyTorch models.
#
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):

net = Net()
net.load_state_dict(torch.load(PATH))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images)

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

net.to(device=device)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images=images.to(device=device)
        labels=labels.to(device=device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

net.to(device=device)
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images=images.to(device=device)
        labels=labels.to(device=device)
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

########################################################################
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor onto the GPU, you transfer the neural
# net onto the GPU.
#
# Let's first define our device as the first visible cuda device if we have
# CUDA available:

# Assuming that we are on a CUDA machine, this should print a CUDA device:

########################################################################
# The rest of this section assumes that ``device`` is a CUDA device.
#
# Then these methods will recursively go over all modules and convert their
# parameters and buffers to CUDA tensors:
#
# .. code:: python
#
#     net.to(device)
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# .. code:: python
#
#         inputs, labels = data[0].to(device), data[1].to(device)
#
# Why dont I notice MASSIVE speedup compared to CPU? Because your network
# is really small.
#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
#
# Training on multiple GPUs
# -------------------------
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out :doc:`data_parallel_tutorial`.
#
# Where do I go next?
# -------------------
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train a face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train a face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: https://pytorch.slack.com/messages/beginner/



```


#### 7.4采用GPU后的代码运行结果

[![rUmu2F.md.png](https://s3.ax1x.com/2020/12/19/rUmu2F.md.png)](https://imgchr.com/i/rUmu2F)



#### 7.5实验结果分析

我们在GPU版本中尝试增加网络的宽度（第一个参数2`nn.Conv2d`和第二个参数1 `nn.Conv2d`–它们必须是相同的数字），看看会得到什么样的加速。

但是我们发现，有无GPU的版本在运行时间上几乎相差无几，GPU版本仅仅是在准确率上稍微优于无GPU版本。

可能的原因是：我们发现运行过程中，由于该网络过于简单，且GPU的内存占有率并不高，导致这两个版本并没有体现出明显的差异。