import torch
import torch.nn as nn
import torch.nn.functional as F

# 写成一个固定的残差块


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


# 写成一个固定网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(4, 64, 7, 2, 3)
        # 定义conv1函数的是图像卷积函数：输入2个channel,输出为64个channel, 卷积核为7x7正方形,strides=2,padding经计算为3
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        # 定义conv2函数的是图像卷积函数：输入64个channel,输出为128个channel, 卷积核为3x3正方形,strides=2,paddomh=1
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        # 定义conv3函数的是图像卷积函数：输入128个channel,输出为256个channel, 卷积核为5x5正方形,strides=2,padding=1
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
