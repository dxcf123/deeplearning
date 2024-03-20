import function

path = r''  # 数据集路径，如果该路径下没有MNIST数据集，会自动下载
input_ = 28 * 28  # 输入
hidden = 256  # 隐藏层
output_ = 10  # 输出
num = 1  # 隐藏层数量减1，即当num=0时，代表1层隐藏层
lr = 0.01  # 学习率
device = 'cuda'  # 训练设备
epochs = 30  # 训练次数

# 返回模型，可以进行后处理
model = function.train(path, input_, epochs=5, device='cuda')
