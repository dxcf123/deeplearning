import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import functional as F
import classmodel
import torch.nn as nn


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
    mnist_train = torchvision.datasets.CIFAR10(  # 加载MNIST数据集，如果本地没有会自动下载
        root=path, train=True, transform=transform, download=True)
    # 测试集
    mnist_test = torchvision.datasets.CIFAR10(
        root=path, train=False, transform=transform, download=True)

    # 创建dataloader对象
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def image2embed(image, patch_size):
    """
    将图像转换为嵌入向量
    :param image: 图片  batch_size * channel * h * w
    :param patch_size: 块大小
    :return:
    """
    patch = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(-1, -2)  # 将图片分成块，它实质是将卷积的部分直接取出来
    return patch  # 将块映射到嵌入向量


def testacc(model, test, epoch, device):
    """
    测试准确率
    :param model: 模型
    :param test: 测试集
    :param epoch: 第epoch轮
    :param device: 设备
    :return:
    """
    all = 0  # 样本总数
    right = 0  # 正确个数
    model.eval()
    for i, (data, label) in enumerate(test):
        all += 32
        data = data.to(device)
        label = label.to(device)
        pre = model(data)[:, 0, :]
        pre = torch.argmax(pre, dim=-1)  # 获取最大值标签
        right += (pre == label).sum()  # 统计每轮正确的数量
    print(epoch, right / all)


def train(path, batchsize, patchsize, emb_dim=512, head=64, device='cpu', lr=0.0001, N=6):
    """
    训练模型
    :param path: 数据集路径
    :param batchsize: 批量大小
    :param patchsize: 块大小
    :param emb_dim: 嵌入纬度
    :param head: 多头
    :param device: 设备
    :param lr: 学习率
    :param N: Encoder层数
    :return: 模型
    """
    train, test = get_dataset(r'C:\Users\30535\Desktop\CodeProgram\Python\deepstudy\data', batchsize)
    # 损失函数
    lossf = nn.CrossEntropyLoss()

    # 用于位置编码的一个参数，它的大小等于  图片通道数 * (一张图片一行数据的大小//patchsize)²
    psize = (32 // patchsize) * (32 // patchsize) + 1
    channel = 3  # 图片通道数

    # 创建VIT模型
    model = classmodel.VIT(channel, batchsize, psize, patchsize, emb_dim, head, device, N=N)
    # 设置优化器
    optm = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    for epo in range(100):
        model.train()
        for i, (data, label) in enumerate(train):
            data = data.to(device)
            label = F.one_hot(label, 10)
            label = label.to(device)
            optm.zero_grad()
            pre = model(data)[:, 0, :]
            loss = lossf(pre, label.float())
            loss.backward()
            optm.step()
        testacc(model, test, epo, device)
    return model
