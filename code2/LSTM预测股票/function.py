import os

# 设置环境变量 KMP_DUPLICATE_LIB_OK 为 TRUE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import classmodel
import torch
from torch.utils.data import DataLoader
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt


def pre_process(path):
    """
    读取我大A股票数据，并进行预处理
    :param path: A股股票数据路径
    :return: 返回预处理后的数据
    """
    # 读取股票数据
    data = pd.read_csv(path)
    # 获取开盘价与收盘价的差值
    bais = max(data['收盘']) - min(data['收盘'])
    data['收盘'] = data['收盘'] / bais  # 数据归一化
    return data, bais


def split_data(stock, lookback, late=1):
    """
    将数据集划分为训练集和测试集
    :param stock: 原始股票数据
    :param lookback: 使用前 lookback天 的数据作为输入，本项目使用前10天的数据作为输入
    :param late: 预测late天后的股票价格，默认1天后
    :return: 返回训练集和测试集
    """
    # 数据集制作
    data_raw = stock['收盘'].to_numpy()  # 将股票的收盘价转换为numpy
    data = []  # 用于保存划分的数据集
    #  制作与划分数据集
    for index in range(len(data_raw) - lookback - late + 1):
        data.append(data_raw[index + late - 1: index + late - 1 + lookback])
    # 获得相应标签
    labels = data_raw[lookback+late-1:]

    # 随机打乱数据集
    index = [i for i in range(int(len(data) * 0.90))]
    random.shuffle(index)
    # 90%的数据作为训练集，5%的作为测试集，5%的数据作为验证集
    train_index = index[:int(len(data) * 0.90)]

    train_x = [data[i] for i in train_index]
    train_y = [labels[i] for i in train_index]
    test_x = data[int(len(data) * 0.90):int(len(data) * 0.95)]
    test_y = labels[int(len(data) * 0.90):int(len(data) * 0.95)]

    val_x = data[int(len(data) * 0.95):]
    val_y = labels[int(len(data) * 0.95):]

    return train_x, train_y, test_x, test_y, val_x, val_y


def train(path, late, input_dim=1, hidden_dim=64, num_layer=2, output_dim=1, epochs=500, batch_size=64):
    """
    训练模型
    :return:
    """
    # 获取我大A的股票数据以及极差
    data, bais = pre_process(path)
    # 数据集划分
    train_x, train_y, test_x, test_y, val_x, val_y = split_data(data, 30, late)

    # 将数据集转换为torch的tensor格式
    # 训练集
    train_x = torch.tensor(train_x, dtype=torch.float32).unsqueeze(-1)
    train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1)
    # 测试集
    test_x = torch.tensor(test_x, dtype=torch.float32).unsqueeze(-1)
    test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(-1)
    # 验证集
    val_x = torch.tensor(val_x, dtype=torch.float32).unsqueeze(-1)
    val_y = torch.tensor(val_y, dtype=torch.float32).unsqueeze(-1)

    # 创建模型
    model = classmodel.Lstm(input_size=input_dim, hidden_dim=hidden_dim, num_layers=num_layer, output_dim=output_dim)
    # 损失函数
    lossF = nn.MSELoss()
    # 创建DataSet
    train_data = classmodel.myDataset(train_x, train_y)
    trst_data = classmodel.myDataset(test_x, test_y)
    val_data = classmodel.myDataset(val_x, val_y)
    # 创建DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(trst_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to('cuda')
    # 用于保存最佳模型的误差
    acc = 10000
    # 训练模型
    for epoch in range(epochs):
        model.train()
        for j, (x, y) in enumerate(train_loader):
            x = x.to('cuda')
            y = y.to('cuda')
            pred = model(x)
            loss = lossF(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 模型评估
        with torch.no_grad():
            model.eval()
            d1 = []  # 用于保存预测值
            y1 = []  # 用于保存真实值
            for j, (x, y) in enumerate(test_loader):
                x = x.to('cuda')
                y = y.to('cuda')
                pred = model(x)
                d1.extend(pred.cpu().numpy() * bais)
                y1.extend(y.cpu().numpy() * bais)
            # 计算预测值与真实值的误差
            loss = lossF(torch.tensor(d1[:50], dtype=torch.float32), torch.tensor(y1[:50], dtype=torch.float32))
            # 模型保存
            if loss < acc:
                acc = loss
                torch.save(model.state_dict(), f'./models2/A{late}.pt')
                print(late, acc)
    # 模型评估
    with torch.no_grad():
        weights = torch.load(f'./models2/A{late}.pt')
        # 加载模型
        model = classmodel.Lstm(input_size=input_dim, hidden_dim=hidden_dim, num_layers=num_layer,
                                output_dim=output_dim)
        model.load_state_dict(weights)
        model = model.to('cuda')
        model.eval()
        d1 = []
        y1 = []
        for j, (x, y) in enumerate(val_loader):
            x = x.to('cuda')
            y = y.to('cuda')
            pred = model(x)
            d1.extend(pred.cpu().numpy() * bais)
            y1.extend(y.cpu().numpy() * bais)
        #  计算预测值与真实值的误差
        loss = lossF(torch.tensor(d1, dtype=torch.float32), torch.tensor(y1, dtype=torch.float32))
        print(late, loss)

        # 绘制预测值与真实值
        # l = [i for i in range(len(d1))]
        #
        # plt.plot(l, d1, label='predict')
        # plt.plot(l, y1, label='true')
        # plt.legend()
        # plt.show()
    return model
