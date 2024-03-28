from torch import nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        残差块
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        """
        super(ResnetBlock, self).__init__()
        # 卷积和规范化层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):  # 卷积运算的前向传播
        x_ = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # 这里1*1卷积需要对输入改变形状，它只支持2D和3D的
        x_ = self.conv3(x_.view(x_.shape[0], x_.shape[1], -1))
        x_ = x_.view(x_.shape[0], x_.shape[1], 16, 16)
        return F.relu(x + x_)


class ResNet(nn.Module):
    def __init__(self, n):
        """
        ResNet模型
        :param n: 卷积块的个数
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = nn.ModuleList([ResnetBlock(64, 64) for _ in range(n)])
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.l1 = nn.Linear(64 * 64, 10)

    def forward(self, x):  # 前向传播
        x = self.pool(F.relu(self.conv1(x)))
        for block in self.block:
            x = block(x)
        x = self.pool2(x)
        x = x.flatten(1)
        x = self.l1(x)
        return x
