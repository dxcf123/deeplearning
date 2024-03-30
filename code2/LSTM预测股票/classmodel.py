import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class myDataset(Dataset):  # 创建数据类
    def __init__(self, data, labels):  # 初始化
        """
        :param data:  原始数据
、       :param labels:  真实值
        """
        self.data = data
        self.labels = labels

    def __getitem__(self, index):  # 获取一条数据并处理
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):  # 返回数据集大小
        return len(self.data)


class Lstm(nn.Module):  # 构建模型
    def __init__(self, input_size, hidden_dim, output_dim, num_layers):
        # input_size 输入维度，hidden_dim 隐藏层维度，batch_first 是否将batch_size的放在第一个维度 bidirectional 是否使用双向lstm，num_layers 层数

        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=False)

        self.dropout = nn.Dropout(0.2)  # 随机失活

        self.flatten = nn.Flatten(0, 1)  # 第0和第1维度进行展平

        # output_dim 输出维度
        self.linear = nn.Linear(hidden_dim, output_dim)

        self.loss = nn.CrossEntropyLoss()  # 损失函数

        self.hidden_num = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, h0=None, c0=None):  # 前向传播
        x = x.to('cuda')
        if h0 is None or c0 is None:
            # h0与c0 的形状是(num_layers, batch_size, hidden_dim) 输入x的第0维就是batch_size
            h0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_num), dtype=torch.float32)
            c0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_num), dtype=torch.float32)
        h0 = h0.to('cuda')
        c0 = c0.to('cuda')
        x, (h0, c0) = self.lstm(x, (h0, c0))  # 当不提供h0,c0时，默认是0
        x = self.linear(x[:, -1, :])
        return x
