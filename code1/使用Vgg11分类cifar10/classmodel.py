import torch
from torch import nn


class VggBlock(nn.Module):
    """
    VGG网络的基本模块
    """

    def __init__(self, in_channels, out_channels, num_convs=1):
        super(VggBlock, self).__init__()
        """
        定义VGG网络的基本模块
        :param in_channels: 输入特征的通道数
        :param out_channels: 输出特征的通道数
        :param num_covs: 卷积层数量
        """
        super(VggBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.convs = nn.ModuleList(  # 设置卷积层 卷积核大小3*3，填充1
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(num_convs - 1)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最后池化层

    def forward(self, x):  # 前向传播函数
        x = self.conv1(x)
        x = torch.relu(x)
        if self.convs:
            for i in self.convs:  # 执行卷积层与激活函数
                x = i(x)
                x = torch.relu(x)
        return self.pool(x)  # 执行池化层


class Vgg(nn.Module):
    def __init__(self):
        """
        VGG网络模型
        网络层结构 卷积层→卷积层→池化层→卷积层→卷积层→池化层→卷积层→卷积层→池化层→全连接层→输出
        """
        super(Vgg, self).__init__()
        self.vgg1 = VggBlock(3, 16, 2)  # VGG块
        self.vgg2 = VggBlock(16, 32, 3)
        self.vgg3 = VggBlock(32, 64, 3)

        # 输入：将经历VGG块的输出展平乘以输出通道数
        self.linear1 = nn.Linear(64 * 64, 64 * 64)  # 线性层
        self.linear2 = nn.Linear(64 * 64, 64 * 64)  # 线性层
        self.linear3 = nn.Linear(64 * 64, 10)  # 线性层

    def forward(self, x):  # 前向传播
        x = self.vgg1(x)
        x = self.vgg2(x)
        x = self.vgg3(x)
        x = self.linear1(x.view(-1, 64 * 64))
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)

        x = torch.softmax(x, dim=1)
        return x
