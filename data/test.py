from datetime import datetime, timedelta
import requests
import pandas as pd
import torch
import numpy as np
from torch import nn
import influxdb_client, time
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        sequences.append(sequence)
    return np.array(sequences)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # num_layers 表示LSTM层数
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


while True:
    # 获取当前日期
    now = datetime.today().date()
    # 计算第二天的日期
    next_day = now + timedelta(days=1)

    # 设置时间为早上九点
    next_day_nine_am = datetime(next_day.year, next_day.month, next_day.day, 9, 0, 0)

    # 获取时间戳
    timestamp = int(next_day_nine_am.timestamp())
    now_t = time.time()
    while now_t < timestamp:
        time.sleep(120)
        now_t = time.time()
    headers = {
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'Connection': 'keep-alive',
        'Host': 'graph.weatherdt.com.cn',
        'Referer': 'http://www.weather.com.cn/',
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36 Edg/122.0.0.0'
    }

    res = requests.get(
        f'http://graph.weatherdt.com.cn/pollen/hfindex.html?eletype=1&city=beijing&start={next_day}&end={next_day}&callback=callback&_={int(time.time() * 1000)}',
        headers=headers)

    data = res.content.decode('utf-8')
    data = eval(data[9:-1])
    index = data['dataList'][0]['num']
    pds = pd.read_csv('./data/huafen.csv')
    data = pd.concat([pds, pd.DataFrame({'date': next_day, 'index': index}, index=[0])], axis=0, ignore_index=True)
    data.to_csv('./data/huafen.csv', index=False)
    data[data['index'] < 0] = 0
    a = max(data['index'])
    data['index'] = data['index'] / a
    seq_length = 7
    input_size = 1
    # 创建输入数据和目标数据
    sequences = create_sequences(data['index'].values, seq_length)
    X_train = torch.Tensor(sequences).view(-1, seq_length, input_size)
    y_train = torch.Tensor(data['index'][seq_length:].values).view(-1, 1)

    # 划分训练集和测试集

    input_size = 1
    hidden_size = 50
    num_layers = 1
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 500
    model.train()
    for epoch in range(num_epochs):
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    data2 = list(data['index'][-7:])

    pre = model(torch.tensor(data2).reshape(-1, 7, 1))  # 预测值
    while pre < 0:
        pre = 0
    pre *= a

    token = "bUrtmbB1kh6JJa4n3Jz74zLsjf1HTbD6OAskmc3ZsUoecD_PtMQhTNfcjCsXB2w9h0x0D7dVPphG3NFQsZj2yg=="

    org = "huafen"
    url = "http://localhost:8086"

    write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
    bucket = "huafen"
    write_api = write_client.write_api(write_options=SYNCHRONOUS)
    point2 = (Point("index").field("index", float(index)))
    write_api.write(bucket=bucket, org="Huafen", record=point2)


    write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
    write_api = write_client.write_api(write_options=SYNCHRONOUS)
    point = (Point("pre_index").field("pre_index", float(pre[0][0])))
    write_api.write(bucket=bucket, org="Huafen", record=point)
