import torch
from torch import nn


class Model(nn.Module):  # 构建全连接层网络
    def __init__(self, input_, hidden, output_, num):
        """
        :param input_: 输入维度
        :param hidden: 隐藏维度
        :param output_: 输出维度
        :param num: 隐藏层数量减1，本网络至少1层隐藏层，即当num=0时，代表1层隐藏层
        """
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_, hidden)  # 第一层
        self.fc2 = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(num)])  # 中间层 ,这里必须使用ModuleList进行封装，否则无法迁移到GPU上
        self.fc3 = nn.Linear(hidden, output_)

    def forward(self, x):  # 线形层→激活函数→线形层→激活函数→线形层
        x = torch.relu(self.fc1(x))
        for i in self.fc2:
            x = torch.relu(i(x))
        x = self.fc3(x)
        return x
