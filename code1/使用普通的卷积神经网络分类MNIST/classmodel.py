import torch
from torch import nn


class Model(nn.Module):  # 构建全连接层网络
    def __init__(self):
        super(Model, self).__init__()
        # 输入通道，输出通道，卷积核大小，步长，填充
        self.cov1 = nn.Conv2d(1, 10, 3, stride=1, padding=1)
        self.cov2 = nn.Conv2d(10, 10, 3, stride=1, padding=1)
        # 池化层 核大小2*2，步长2
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        # 线性层
        self.lin1 = nn.Linear(490, 10)

    def forward(self, x):  # 线形层→激活函数→线形层→激活函数→线形层
        x = self.cov1(x)  # 第一层卷积 28→28
        x = self.maxpool1(x)  # 第一层池化 28→14
        x = torch.relu(x)  # 激活函数
        x = self.cov2(x)  # 第二层卷积 14→14
        x = self.maxpool2(x)  # 第二层池化 14→7
        x = torch.relu(x)  # 激活函数
        x = x.view(x.size(0), -1)  # 将特征展平 10*7*7→490
        x = self.lin1(x)  # 全连接层 490→10
        return x
