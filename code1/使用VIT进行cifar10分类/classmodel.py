import torch
import torch.nn as nn
import function
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, channel, batchsize, psize, patchsize, emb_dim, device):
        """
        词嵌入层
        :param batchsize: 批量大小
        :param psize: 用于位置编码的一个参数，它的大小等于  图片通道数 * (一张图片一行数据的大小//patchsize)²
        :param patchsize: 提取图块的边长
        :param emb_dim: 嵌入维度
        :param device: 运算设备
        """
        super(Embedding, self).__init__()
        self.pathF = function.image2embed  # 导入提取图片块的函数
        self.patchszie = patchsize  # 边长
        self.emb_dim = emb_dim  # 嵌入纬度
        self.l1 = nn.Linear(patchsize * patchsize * channel, emb_dim)  # 用于将图片块映射为为嵌入纬度大小
        # 定义一个矩阵嵌入到输入数据开头，表示数据的开始
        self.cls_token_emb = torch.randn(batchsize, 1, self.emb_dim, requires_grad=True, device=device)
        # 位置编码
        self.position_emb = torch.randn(1, psize, self.emb_dim, requires_grad=True, device=device)

    def forward(self, x):  # 前向传播
        """
        这里将图片块转换为嵌入纬度，加入了开头与位置编码
        :param x:
        :return:
        """

        x = self.pathF(x, self.patchszie)
        x = self.l1(x)
        # print(x.shape)
        # print(self.cls_token_emb.shape)
        x = torch.cat((self.cls_token_emb[:x.shape[0]], x), dim=1)
        x += self.position_emb
        return x


class Attention(nn.Module):
    def __init__(self, emb_dim=128, head=8):
        """
        注意力机制
        :param emb_dim: 词嵌入纬度
        :param head: 多头头数
        """
        super(Attention, self).__init__()
        assert emb_dim % head == 0  # 保证emb_dim可以整除head，注意力机制的词嵌入维度需要是多头的n倍
        self.emb_dim = emb_dim  # 词嵌入纬度
        self.head = head  # 多头
        self.head_dim = emb_dim // head

        # q k v 三个输入的线性层  维度变换 emb_dim → emb_dim
        self.query_L = nn.Linear(emb_dim, emb_dim)
        self.key_L = nn.Linear(emb_dim, emb_dim)
        self.value_L = nn.Linear(emb_dim, emb_dim)

    def forward(self, q, k, v):
        """
        前向传播 q,k,v为transformer的三个输入，这里做了注意力机制的运算
        :return:
        """
        # q,k,v的形状为 batchsize 长度 词嵌入纬度 ，下面batchsize，长度，词嵌入纬度，头数，分别用 B L D H 代替
        # 这里进行多头注意力机制进行计算，因此需要进行纬度变换
        x_q = self.query_L(q)  # q 进行线性层变换 B,L,D → B,L,D
        x_q = x_q.reshape(q.shape[0], q.shape[1], self.head, self.head_dim)  # B,L,D → B,L,H,D/H
        x_q = x_q.transpose(1, 2)  # B,L,H,D/H → B,H,L,D/H
        x_q = x_q.reshape(-1, q.shape[1], self.head_dim)  # B,H,L,D/H  → BH,L,D/H

        # k,v操作与q相同
        x_k = self.key_L(k).reshape(k.shape[0], k.shape[1], self.head, self.head_dim)
        x_k = x_k.transpose(1, 2)
        x_k = x_k.reshape(-1, k.shape[1], self.head_dim)

        x_v = self.value_L(v).reshape(v.shape[0], v.shape[1], self.head, self.head_dim)
        x_v = x_v.transpose(1, 2)
        x_v = x_v.reshape(-1, v.shape[1], self.head_dim)

        # 注意力机制计算，这里需要对x_K进行转置才符合运算规则
        x_k = x_k.transpose(1, 2)  # BH,L,BH  →  BH,D/H,L
        x_atten = torch.matmul(x_q, x_k) / (self.head_dim ** 0.5)  # q,k相乘并除以根号D → BH,L,L
        x_atten = F.softmax(x_atten, dim=-1)

        x_out = torch.matmul(x_atten, x_v)  # → BH,L,D/H
        x_out = x_out.reshape(-1, self.head, x_out.shape[1], x_out.shape[2])  # BH,L,D/H → B,H,L,D/H
        x_out = x_out.transpose(1, 2)  # B,H,L,D/H → B,L,H,D/H
        x = x_out.reshape(-1, x_out.shape[1], self.head * self.head_dim)  # B,L,H,D/H->B,L,D
        return x


class Encoder(nn.Module):
    def __init__(self, psize, emb_dim=128, head=8):
        """
        编码器
        :param psize: 用于位置编码的一个参数，它的大小等于  图片通道数 * (一张图片一行数据的大小//patchsize)²
        :param emb_dim: 嵌入维度
        :param head: 多头头数
        """
        super(Encoder, self).__init__()
        self.Attention = Attention(emb_dim, head)  # 注意力机制
        # 前馈全连接子层
        self.l1 = nn.Linear(emb_dim, emb_dim)
        self.l2 = nn.Linear(emb_dim, emb_dim)
        # 规范化层
        self.norm1 = nn.BatchNorm1d(psize)
        self.norm2 = nn.BatchNorm1d(psize)

    def forward(self, q, k, v):  # 前向传播计算
        # 注意力机制
        x = self.Attention(q, k, v)
        # 规范化层
        x = self.norm1(x + q)
        # 全连接层
        x_ = self.l1(x)
        x_ = F.gelu(x_)
        x_ = self.l2(x_)
        # 规范化层
        x = self.norm2(x + x_)
        return x


class VIT(nn.Module):
    def __init__(self, channel, batchsize, psize, patchsize, emb_dim, head, device, N=6):
        """
        VIT模型
        :param batchsize: 批量
        :param psize: 用于位置编码的一个参数，它的大小等于  图片通道数 * (一张图片一行数据的大小//patchsize)²
        :param patchsize: 图片块边长
        :param emb_dim: 嵌入维度
        :param head: 多头
        :param device: 运算设备
        """
        super(VIT, self).__init__()
        self.Embed = Embedding(channel, batchsize, psize, patchsize, emb_dim, device)  # 词嵌入层
        self.Encoder = torch.nn.ModuleList([Encoder(psize, emb_dim, head) for _ in range(N)])
        # 用于分类的全连接层
        self.l1 = nn.Linear(emb_dim, emb_dim)
        self.l2 = nn.Linear(emb_dim, 10)  # CIFAR10 10分类

    def forward(self, x):
        #  词嵌入层
        x = self.Embed(x)
        #  编码器层
        for i in self.Encoder:
            x = i(x, x, x)
        #  分类层
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x
