import torch
import torch.nn as nn
import torch.nn.functional as F

# 写成一个固定的残差块


def Resblock(x):
    out = x
    conv1 = nn.Conv2d(512，512，3，1，1)
    # 8*8*512,kernel=3,strides=1,计算padding=1
    out = conv1(out)
    bn1 = nn.BatchNorm2d(512)
    # outchannel为512
    out = bn1(out)
    out = F.relu(out)
    conv2 = nn.Conv2d(512, 512, 3, 1, 1)
    # 同上
    out = conv2(out)
    bn2 = nn.BatchNorm2d(512)
    out = bn2(out)
    return(F.relu(x+out))  # 传出激活后的值


# 写成一个固定网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(2, 64, 7, 2, 3)
        # 定义conv1函数的是图像卷积函数：输入2个channel,输出为64个channel, 卷积核为7x7正方形,strides=2,padding经计算为3
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        # 定义conv2函数的是图像卷积函数：输入64个channel,输出为128个channel, 卷积核为3x3正方形,strides=2,paddomh=1
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        # 定义conv3函数的是图像卷积函数：输入128个channel,输出为256个channel, 卷积核为5x5正方形,strides=2,padding=1
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        # 同上
        self.fc1 = nn.Linear(8*8*512, 1024)
        # 全联接层
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 21)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # 先卷积在激活
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # 到残差块
        x = Resblock(x)
        x = Resblock(x)
        # 做两次，传出激活后的值
        x = x.view(-1, 8*8*512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
