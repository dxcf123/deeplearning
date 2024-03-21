import torch
import torchvision
from torch.utils.data import DataLoader
import classmodel
from torch.nn import functional as F


def get_dataset(path, batch_size=32, transform=None):
    """
    加载MNIST数据集并将其转换为DataLoader对象。
    :param path: 数据集路径
    :param batch_size: 批处理大小
    :param transform: 数据预处理
    :return: 训练集与测试集的DataLoader对象
    """
    if transform is None:
        transform = torchvision.transforms.Compose([  # 对图像进行预处理
            torchvision.transforms.ToTensor(),  # 将图片转换成张量
            torchvision.transforms.Normalize((0.5,), (0.5,))  # 对图像进行归一化处理
        ])

    # 训练集
    mnist_train = torchvision.datasets.MNIST(  # 加载MNIST数据集，如果本地没有会自动下载
        root=path, train=True, transform=transform, download=True)
    # 测试集
    mnist_test = torchvision.datasets.MNIST(
        root=path, train=False, transform=transform, download=True)

    # 创建dataloader对象
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def acc_test(loader, model, device):
    """
    计算模型在测试集上的准确率。
    :param loader: 测试集的DataLoader对象
    :param model: 模型对象
    :param device: 设备对象
    :return: 准确率
    """
    model.eval()  # 将模型设置为评估模式
    acc = 0  # 准确的个数
    all_ = 0  # 总个数
    with torch.no_grad():  # 不计算梯度
        for i, (x, y) in enumerate(loader):  # 获取输入与输出
            x = x.to(device)  # 将图片转换为一维张量
            y = y.to(device)
            pre = model(x)  # 预测
            pre = torch.argmax(pre, dim=1)  # 获取预测结果每行中的最大值的坐标
            all_ += len(pre)  # 记录数据总数
            acc += (pre == y).sum().item()  # 记录准确的个数
    return acc / all_  # 返回准确率


def train(path, output_=10, batch_size=32, lr=0.01, device='cpu', epochs=30):
    """
    训练模型
    :param path: 数据存放路径
    :param input_: 输入神经元个数
    :param hidden: 隐藏层神经元个数
    :param output_: 输出层神经元个数
    :param num: 隐藏层数量减1，即当num=0时，代表1层隐藏层
    :param lr: 学习率
    :param device: 训练设备
    :param epochs: 训练轮数
    :return: 返回训练后的模型
    """
    # 损失函数设置为交叉熵损失
    lossFuction = torch.nn.CrossEntropyLoss()

    # 创建一个全连接网络的对象
    model = classmodel.Model()

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 使用Adam优化器

    # 获取数据
    train_loader, test_loader = get_dataset(path, batch_size=batch_size)

    # 将模型移动到设备上
    model.to(device)

    # 模型设置为训练模式
    model.train()

    # 训练模型
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):  # 获取输入与输出
            x = x.to(device)  # 将图片转换移动到设备上
            # 将输出数据转换为one_hot编码并转换为32位浮点数并移动到设备上
            y = F.one_hot(y, num_classes=10).to(device)
            optimizer.zero_grad()  # 将优化器梯度置零
            pre = model(x)  # 预测数据
            loss = lossFuction(pre, y.float())  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新
            if (i + 1) % 100 == 0:
                print(acc_test(test_loader, model, device))
                model.train()
    return model
