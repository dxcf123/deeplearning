import function

path = r'C:\Users\30535\Desktop\CodeProgram\Python\deepstudy\data'  # 数据路径
batch_size = 128  # 批量
lr = 0.0001  # 学习率
device = 'cuda'  # 训练设备
epochs = 10
model = function.train(path, batch_size, lr, device, epochs)
