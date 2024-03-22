import function

path = r''  # 数据路径
batch_size = 64  # 批量
lr = 0.0001  # 学习率
device = 'cuda'  # 训练设备
epochs = 10
model = function.train(path, batch_size, lr, device, epochs)
