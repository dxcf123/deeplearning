import torch
from torch import nn
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, ch_data, en_data, ch_word_2_index, en_word_2_index, device):
        """
        创建数据集
        :param ch_data: 中文数据
        :param en_data: 英文数据
        :param ch_word_2_index: 中文字符对应编码
        :param en_word_2_index: 英文字符对应编码
        :param device: 训练设备
        """
        self.ch_data = ch_data
        self.en_data = en_data
        self.ch_word_2_index = ch_word_2_index
        self.en_word_2_index = en_word_2_index
        self.device = device

    def __getitem__(self, index):
        en = self.en_data[index]
        ch = self.ch_data[index]
        en_index = [self.en_word_2_index[i] for i in en]  # 将英文单词转换为索引
        ch_index = [self.ch_word_2_index[i] for i in ch]  # 将中文单词转换为索引
        return en_index, ch_index

    def __len__(self):
        return len(self.ch_data)

    def another_process(self, batch_datas):
        """
        特殊处理，这里传入一个batch的数据，并对这个batch的数据进行填充，使得每一行的数据长度相同。这里填充pad 空字符  bos 开始  eos结束
        :param batch_datas: 一个batch的数据
        :return: 返回填充后的数据
        """
        # 创建四个空字典存储数据
        en_index, ch_index = [], []  # 中文英文索引，中文索引
        en_len, ch_len = [], []  # 没行英文长度，每行中文长度

        for en, ch in batch_datas:  # 对batch进行遍历，将所有数据的索引与长度加入四个列表
            en_index.append(en)
            ch_index.append(ch)
            en_len.append(len(en))
            ch_len.append(len(ch))

        # 获取中英文的最大长度，根据这个长度对所有数据进行填充，使每行数据长度相同
        max_en_len = max(en_len)
        max_ch_len = max(ch_len)

        # 英文数据填充，i是原始数据，后面是填充的pad
        en_index = [i + [self.en_word_2_index['<pad>']] * (max_en_len - len(i)) for i in en_index]
        # 中文数据填充 先填充bos表示句子开始，后面接原始数据，最后填充eos表示句子结束，后面接pad
        ch_index = [[self.ch_word_2_index['<bos>']] + i + [self.ch_word_2_index['<eos>']] +
                    [self.ch_word_2_index['<pad>']] * (max_ch_len - len(i)) for i in ch_index]

        # 将处理后的数据转换为tensor并放到相应设备上
        en_index = torch.tensor(en_index, device=self.device)
        ch_index = torch.tensor(ch_index, device=self.device)
        return en_index, ch_index


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_num, en_len):
        """
        创建编码器 这里用于英文部分的LSTM层
        :param embedding_dim: 词嵌入纬度
        :param hidden_num: 这个用于lstm的隐藏层纬度
        :param en_len: 英语词语统计的个数，用于创建词嵌入层
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(en_len, embedding_dim)  # 创建词嵌入层
        self.lstm = nn.LSTM(embedding_dim, hidden_num, batch_first=True)  # 创建LSTM层

    def forward(self, x):
        """
        前向传播 这里不关心英文部分的输出，仅仅得到lstm的h0与c0作为中文部分LSTM的初试状态
        :param x: 一则输入数据
        :return:
        """
        embedded = self.embedding(x)  # 词嵌入
        _, (h0, c0) = self.lstm(embedded)  # 英文部分lstm
        return h0, c0


class Dncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_num, ch_len):
        """
        创建解码器 这里用于中文部分的LSTM层
        :param embedding_dim: 词嵌入纬度
        :param hidden_num: 这个用于lstm的隐藏层纬度 这里的隐层需要和Encoder部分的相同，因为会传入它的h0,c0进行矩阵运算。
        :param ch_len: 中文词语统计的个数，用于创建词嵌入层
        """
        super(Dncoder, self).__init__()
        self.embedding = nn.Embedding(ch_len, embedding_dim)  # 创建词嵌入层
        self.lstm = nn.LSTM(embedding_dim, hidden_num, batch_first=True)  # 创建LSTM层

    def forward(self, x, h0, c0):
        """
        # 中文部分前向传播
        :param x:
        :param h0: Encoder层的输出
        :param c0: Encoder层的输出
        :return:
        """
        x = self.embedding(x)  # 词嵌入
        x, (h0, c0) = self.lstm(x, (h0, c0))  # LSTM层
        return x, h0, c0


class Seq2Seq(nn.Module):
    def __init__(self, e_embedding_dim, hidden_num, en_len, d_embedding_num, ch_len):
        """
        创建seq2seq模型 包括Encoder和Decoder
        :param e_embedding_dim: Encoder词嵌入纬度
        :param hidden_num: 隐藏层
        :param en_len: 英文词汇统计的个数
        :param d_embedding_num: Decoder词嵌入纬度
        :param ch_len: 中文词汇统计的个数
        """
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(e_embedding_dim, hidden_num, en_len)  # 创建Encoder
        self.decoder = Dncoder(d_embedding_num, hidden_num, ch_len)  # 创建Decoder
        # 创建线性层，decoder的输出纬度等于词嵌入纬度，这里将其投射为中文字符长度的数组，数组值最大的所有即为预测的字符的索引
        self.linear = nn.Linear(hidden_num, ch_len)
        self.loss = nn.CrossEntropyLoss()  # 损失函数使用交叉熵

    def forward(self, en_index, ch_index):
        """
        前向传播
        :param en_index: 转换为索引后的英文数据
        :param ch_index: 转换为索引后的中文数据
        :return:
        """
        # LSTM常规操作,不懂就去看LSTM
        decoder_in = ch_index[:, :-1]
        label = ch_index[:, 1:]

        h0, c0 = self.encoder(en_index)  # 获取Encoder的 h0,c0作为decoder的输入
        x, _, _ = self.decoder(decoder_in, h0, c0)  # 获取Decoder的输出
        pre = self.linear(x)  # 将Decoder的输出投射到中文字符长度的空间
        # 计算损失
        loss = self.loss(pre.contiguous().view(-1, pre.shape[-1]), label.contiguous().view(-1))
        return loss  # 返回损失
