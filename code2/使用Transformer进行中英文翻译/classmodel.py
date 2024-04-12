import torch.nn.functional as F
import math
import torch
from nltk.tokenize import word_tokenize
import jieba
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import function
import copy


class TranslationDataset(Dataset):
    # 创建数据集
    def __init__(self, src, tgt):
        """
        初始化
        :param src: 源数据(经tokenizer处理后)
        :param tgt: 目标数据(经tokenizer处理后)
        """
        self.src = src
        self.tgt = tgt

    def __getitem__(self, i):
        return self.src[i], self.tgt[i]

    def __len__(self):
        return len(self.src)


class Tokenizer:
    ## 定义tokenizer,对原始数据进行处理
    def __init__(self, en_path, ch_path, count_min=5):
        """
        初始化
        :param en_path: 英文数据路径
        :param ch_path: 中文数据路径
        :param count_min: 对出现次数少于这个次数的数据进行过滤
        """
        self.en_path = en_path  # 英文路径
        self.ch_path = ch_path  # 中文路径
        self.__count_min = count_min  # 对出现次数少于这个次数的数据进行过滤

        # 读取原始英文数据
        self.en_data = self.__read_ori_data(en_path)
        # 读取原始中文数据
        self.ch_data = self.__read_ori_data(ch_path)
        # 英文index_2_word
        self.en_index_2_word = ['0']
        # 中文index_2_word
        self.ch_index_2_word = ['0']
        # 英文word_2_index
        self.en_word_2_index = {'unK': 0}
        # 中文word_2_index
        self.ch_word_2_index = {'unK': 0}
        # 中英文字符计数
        self.__en_count = {'<pad>': count_min}
        self.__ch_count = {'<pad>': count_min, '<bos>': count_min, '<eos>': count_min}
        # 批量tokenize数据的开始位置与结束位置，即每次tokenize 30000组数据
        self.__start = 0
        self.__end = 30000
        # 创建英文词汇表
        self.__build_vocab()

    def __read_ori_data(self, path):
        """
        读取原始数据
        :param path: 数据路径
        :return: 返回一个列表，每个元素是一条数据
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read().split('\n')[:-1]
        return data

    def __en_token_count(self):
        """
        英文token计数
        :return:
        """
        x = 0
        # 对英文数据进行遍历
        for sentence in self.en_data:
            x += 1
            if x % 10000 == 0:
                print('en ', x / len(self.en_data))
            # 对句子进行分词
            words = word_tokenize(sentence)
            # 对分词后的结果进行计数
            for word in words:
                if word not in self.__en_count:
                    self.__en_count[word] = 1
                else:
                    self.__en_count[word] += 1

    def __ch_token_count(self):
        """
        中文token计数
        :return:
        """
        x = 0
        #  对中文数据进行遍历
        for sentence in self.ch_data:
            x += 1
            if x % 10000 == 0:
                print('ch ', x / len(self.ch_data))
            #  对句子进行分词
            words = jieba.cut(sentence)
            #   对分词后的结果进行计数
            for word in words:
                if word not in self.__ch_count:
                    self.__ch_count[word] = 1
                else:
                    self.__ch_count[word] += 1

    def __build_vocab(self):
        """
        构建词汇表
        :return:
        """
        # 英文token计数
        self.__en_token_count()
        # 中文token计数
        self.__ch_token_count()
        # 对英文数据的键值对进行遍历
        for word, count in self.__en_count.items():
            # 如果出现次数大于等于最小限度
            if count >= self.__count_min:
                # 加入词表
                self.en_word_2_index[word] = len(self.en_index_2_word)
                self.en_index_2_word.append(word)
            # 否则将这个词对应的index设为0，即指向unk,即当句子中出现时，用unk填充
            else:
                self.en_word_2_index[word] = 0
        self.en_word_2_index.update({'<pad>': len(self.en_index_2_word)})
        self.en_index_2_word.append('<pad>')

        # 释放英文字符的空间
        self.__en_count = None
        # 对中文数据的键值对进行遍历，以下操作与上面相同
        for word, count in self.__ch_count.items():
            if count >= self.__count_min:
                self.ch_word_2_index[word] = len(self.ch_index_2_word)
                self.ch_index_2_word.append(word)
            else:
                self.ch_word_2_index[word] = 0
        self.__ch_count = None
        self.ch_word_2_index.update({'<pad>': len(self.ch_index_2_word), '<bos>': len(self.ch_index_2_word) + 1,
                                     '<eos>': len(self.ch_index_2_word) + 2})
        self.ch_index_2_word.append('<pad>')
        self.ch_index_2_word.append('<bos>')
        self.ch_index_2_word.append('<eos>')

    def split_data(self, data, func):
        data_type = type(data)  # 检测数据类型
        # 判断数据是不是字符串，如果是则放到列表以内
        if data_type == str:
            data = [data]
        # 用于存储编码后的数据
        tokens_data = []
        # 对数据进行遍历
        for sentence in data:
            # 对数据进行分词
            tokens = func(sentence)
            tokens_data.append(list(tokens))
        return tokens_data

    def en_encode(self, data):
        """
        英文数据编码
        :param data: 需要编码的数据
        :return: 返回编码后的数据集
        """
        src = self.split_data(data, word_tokenize)
        tokenized_data = []
        for sentence in src:
            # 用于存放每个句子对应的编码
            en_tokens = []
            # 对分词结果进行遍历
            for i in sentence:
                # 对于结果进行编码
                en_tokens.append(self.en_word_2_index.get(i, 0))
            tokenized_data.append(en_tokens)
            # 返回编码后的数据
        return tokenized_data

    def decode(self, data):
        """
        数据解码
        :param data: 这里传入一个中文的index
        :return: 返回解码后的一个字符
        """
        return self.ch_index_2_word[data]

    def ch_encode(self, data):
        """
        中文编码
        :param data: 需要编码的数据
        :return:
        """
        tgt = self.split_data(data, jieba.cut)
        # 用于存储编码后的数据
        tokenized_data = []
        # 对数据进行遍历
        for sentence in tgt:
            # 用于存放每个句子对应的编码
            ch_tokens = []
            # 对分词结果进行遍历
            for i in sentence:
                # 编码
                ch_tokens.append(self.ch_word_2_index.get(i, 0))
            tokenized_data.append(ch_tokens)
        # 返回编码后的数据
        return tokenized_data

    def __get_datasets(self):
        """
        获取数据集
        :return:返回DataSet类型的数据 或者 None
        """
        # 获取一部分数据，这是一个生成器
        src, tgt = next(self.__data_generator__())
        # 将下一个数据切分位置向后偏移30000
        self.__start += 30000
        self.__end += 30000
        # 如果返回空列表或者None，返回None
        if src == [] or src == None:
            return None
        # 将数据编码并
        src = self.en_encode(src)
        tgt = self.ch_encode(tgt)
        # 返回数据集
        return TranslationDataset(src, tgt)

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
        max_len = max(max_en_len, max_ch_len + 2)

        # 英文数据填充，i是原始数据，后面是填充的pad
        en_index = [i + [self.en_word_2_index['<pad>']] * (max_len - len(i)) for i in en_index]
        # 中文数据填充 先填充bos表示句子开始，后面接原始数据，最后填充eos表示句子结束，后面接pad
        ch_index = [[self.ch_word_2_index['<bos>']] + i + [self.ch_word_2_index['<eos>']] +
                    [self.ch_word_2_index['<pad>']] * (max_len - len(i) + 1) for i in ch_index]

        # 将处理后的数据转换为tensor并放到相应设备上
        en_index = torch.tensor(en_index)
        ch_index = torch.tensor(ch_index)
        return en_index, ch_index

    def get_dataloader(self, batch_size=32):
        """
        获取dataloader
        :return:
        """
        # 获取数据集
        data = self.__get_datasets()
        p = 0
        # 如果数据集为空，返回None
        if data is None:
            self.__start = 0
            self.__end = 30000
            data = self.__get_datasets()
            p = 1
        # 返回DataLoader类型的数据
        return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=self.another_process), p

    def __data_generator__(self):
        # 数据集生成器
        while True:
            # 如果结束位置大于数据集长度，返回剩下的所有数据
            while self.__end > len(self.en_data):
                self.__end = len(self.en_data)
                yield self.en_data[self.__start:self.__end], self.ch_data[self.__start:self.__end]
            # 返回start 到 end之间的源数据
            yield self.en_data[self.__start:self.__end], self.ch_data[self.__start:self.__end]

    # 获取英文词表大小
    def get_en_vocab_size(self):
        return len(self.en_index_2_word)

    # 获取中文词表大小
    def get_ch_vocab_size(self):
        return len(self.ch_index_2_word)

    # 获取数据集大小
    def get_dataset_size(self):
        return len(self.en_data)


class Batch:
    # 批次类,对每一个批次的数据进行掩码生成操作
    def __init__(self, src, trg=None, tokenizer=None, device='cuda'):
        """
        初始化函数
        :param src: 源数据
        :param trg: 目标数据
        :param tokenizer: 分词器
        :param device: 训练设备
        """
        # 将输入、输出单词id表示的数据规范成整数类型并转换到训练设备上
        src = src.to(device).long()
        trg = trg.to(device).long()
        self.src = src  # 源数据 (batch, seq_len)
        self.__pad = tokenizer.ch_word_2_index['<pad>']  # 填充字符的索引
        # 对于当前输入的语句非空部分进行判断，这里是对源数据进行掩码操作，将填充的内容置为0
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != self.__pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对解码器使用的目标语句进行掩码
        if trg is not None:
            # 解码器使用的目标输入部分
            self.trg = trg[:, : -1]
            # 解码器训练时应预测输出的目标结果
            self.trg_y = trg[:, 1:]
            # 将目标输入部分进行注意力掩码
            self.trg_mask = self.make_std_mask(self.trg, self.__pad)
            # 将应输出的目标结果中实际的词数进行统计
            self.ntokens = (self.trg_y != self.__pad).data.sum()

    # 掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """
        生成掩码矩阵
        :param tgt: 目标数据
        :param pad: 填充字符的索引
        :return:
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)  # 首先对pad进行掩码生成
        # 这里对注意力进行掩码操作并与pad掩码结合起来。
        tgt_mask = tgt_mask & Variable(function.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class Embedding(nn.Module):
    # 词嵌入层
    def __init__(self, d_model, vocab):
        """
        词嵌入层初始化
        :param d_model: 词嵌入维度
        :param vocab: 词表大小
        """
        super(Embedding, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x的词向量（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # 位置编码器层
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cuda'):
        """
        位置编码器层初始化
        :param d_model: 词嵌入维度
        :param dropout: dropout比例
        :param max_len: 序列最大长度
        :param device: 训练设备
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 位置编码矩阵，维度[max_len, embedding_dim]
        pe = torch.zeros(max_len, d_model, device=device)
        # 单词位置
        position = torch.arange(0.0, max_len, device=device)
        position.unsqueeze_(1)
        # 使用exp和log实现幂运算
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=device) * (- math.log(1e4) / d_model))
        div_term.unsqueeze_(0)
        # 计算单词位置沿词向量维度的纹理值
        pe[:, 0:: 2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1:: 2] = torch.cos(torch.mm(position, div_term))
        # 增加批次维度，[1, max_len, embedding_dim]
        pe.unsqueeze_(0)
        # 将位置编码矩阵注册为buffer(不参加训练)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个批次中语句所有词向量与位置编码相加
        # 注意，位置编码不参与训练，因此设置requires_grad=False
        x += Variable(self.pe[:, : x.size(1), :], requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    # 多头注意力机制
    def __init__(self, h, d_model, dropout=0.1):
        """
        多头注意力机制初始化
        :param h: 多头
        :param d_model: 词嵌入维度
        :param dropout: dropout比例
        """
        super(MultiHeadedAttention, self).__init__()
        # 确保整除
        assert d_model % h == 0
        # q、k、v向量维数
        self.d_k = d_model // h
        # 头的数量
        self.h = h
        # WQ、WK、WV矩阵及多头注意力拼接变换矩阵WO 4个线性层
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        # 注意力机制函数
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        反向传播
        :param query: q
        :param key: k
        :param value: v
        :param mask: 掩码
        :return:
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        # 批次大小
        nbatches = query.size(0)
        # WQ、WK、WV分别对词向量线性变换，并将结果拆成h块
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 注意力加权
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # 多头注意力加权拼接
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 对多头注意力加权拼接结果线性变换
        return self.linears[-1](x)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        注意力加权
        :param query: q
        :param key: k
        :param value: v
        :param mask: 掩码矩阵
        :param dropout: dropout比例
        :return:
        """
        # q、k、v向量长度为d_k
        d_k = query.size(-1)
        # 矩阵乘法实现q、k点积注意力，sqrt(d_k)归一化
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # 注意力掩码机制
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 注意力矩阵softmax归一化
        p_attn = F.softmax(scores, dim=-1)
        # dropout
        if dropout is not None:
            p_attn = dropout(p_attn)
        # 注意力对v加权
        return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    # 子层连接结构 用于连接注意力机制以及前馈全连接网络
    def __init__(self, d_model, dropout):
        """
        子层连接结构初始化层
        :param d_model: 词嵌入纬度
        :param dropout: dropout比例
        """
        super(SublayerConnection, self).__init__()
        # 规范化层
        self.norm = nn.LayerNorm(d_model)
        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 层归一化
        x_ = self.norm(x)
        x_ = sublayer(x_)
        x_ = self.dropout(x_)
        # 残差连接
        return x + x_


class FeedForward(nn.Module):
    # 前馈全连接网络
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        前馈全连接网络初始化层
        :param d_model: 词嵌入维度
        :param d_ff: 中间隐层维度
        :param dropout: dropout比例
        """
        super(FeedForward, self).__init__()
        # 全连接层
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


#
class Encoder(nn.Module):
    # 编码器
    def __init__(self, h, d_model, d_ff=2048, dropout=0.1):
        """
        编码器层初始化
        :param h: 头数
        :param d_model: 词嵌入维度
        :param d_ff: 中间隐层维度
        :param dropout: dropout比例
        """
        super(Encoder, self).__init__()
        # 多头注意力
        self.self_attn = MultiHeadedAttention(h, d_model)
        # 前馈全连接层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 子层连接结构
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        # 规范化层
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        # attn的结果直接作为下一层输入
        return self.norm(self.sublayer2(x, self.feed_forward))


class Decoder(nn.Module):
    def __init__(self, h, d_model, d_ff=2048, dropout=0.1):
        """
        解码器层
        :param h: 头数
        :param d_model: 词嵌入维度
        :param d_ff: 中间隐层维度
        :param dropout: dropout比例
        """
        super(Decoder, self).__init__()
        self.size = d_model
        # 自注意力机制
        self.self_attn = MultiHeadedAttention(h, d_model)
        # 上下文注意力机制
        self.src_attn = MultiHeadedAttention(h, d_model)
        # 前馈全连接子层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 子层连接结构
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
        # 规范化层
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        # memory为编码器输出隐表示
        m = memory
        # 自注意力机制，q、k、v均来自解码器隐表示
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 上下文注意力机制：q为来自解码器隐表示，而k、v为编码器隐表示
        x = self.sublayer2(x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.norm(self.sublayer3(x, self.feed_forward))


class Generator(nn.Module):
    #  生成器层
    def __init__(self, d_model, vocab):
        """
        生成器层初始化
        :param d_model:
        :param vocab:
        """
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    # Transformer层
    def __init__(self, tokenizer, h=8, d_model=256, E_N=2, D_N=2, device='cuda'):
        """
        transformer层初始化
        :param h: 头数
        :param d_model: 词嵌入纬度
        :param tokenizer:
        :param E_N:
        :param D_N:
        :param device:
        """
        super(Transformer, self).__init__()
        # 编码器
        self.encoder = nn.ModuleList([Encoder(h, d_model) for _ in range(E_N)])
        # 解码器
        self.decoder = nn.ModuleList([Decoder(h, d_model) for _ in range(D_N)])
        # 词嵌入层
        self.src_embed = Embedding(d_model, tokenizer.get_en_vocab_size())
        self.tgt_embed = Embedding(d_model, tokenizer.get_ch_vocab_size())
        # 位置编码器层
        self.src_pos = PositionalEncoding(d_model, device=device)
        self.tgt_pos = PositionalEncoding(d_model, device=device)
        # 生成器层
        self.generator = Generator(d_model, tokenizer.get_ch_vocab_size())

    def encode(self, src, src_mask):
        """
        编码
        :param src: 源数据
        :param src_mask: 源数据掩码
        :return:
        """
        # 词嵌入
        src = self.src_embed(src)
        # 位置编码
        src = self.src_pos(src)
        # 编码
        for i in self.encoder:
            src = i(src, src_mask)
        return src

    def decode(self, memory, tgt, src_mask, tgt_mask):
        """
        解码
        :param memory: 编码器输出
        :param tgt: 目标数据输入
        :param src_mask: 源数据掩码
        :param tgt_mask: 目标数据掩码
        :return:
        """
        #  词嵌入
        tgt = self.tgt_embed(tgt)
        #  位置编码
        tgt = self.tgt_pos(tgt)
        # 解码
        for i in self.decoder:
            tgt = i(tgt, memory, src_mask, tgt_mask)
        return tgt

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        反向传播
        :param src: 源数据
        :param tgt: 目标数据
        :param src_mask: 源数据掩码
        :param tgt_mask: 目标数据掩码
        :return:
        """
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)


class LabelSmoothing(nn.Module):
    # 标签平滑
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        初始化
        :param size: 目标数据词表大小
        :param padding_idx: 目标数据填充字符的索引
        :param smoothing: 做平滑的值，为0即不进行平滑
        """
        super(LabelSmoothing, self).__init__()
        # KL散度，通常用于测量两个概率分布之间的差异
        self.criterion = nn.KLDivLoss(reduction='sum')
        # 目标数据填充字符的索引
        self.padding_idx = padding_idx
        # 置信度
        self.confidence = 1.0 - smoothing
        # 平滑值
        self.smoothing = smoothing
        # 词表大小
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        反向传播
        :param x: 预测值
        :param target: 目标值
        :return:
        """
        # 判断输出值的第二维传长度是否等于输出词表的大小，这里x的shape为 （batch*seqlength,x.shape(-1)）
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 标签平滑填充
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 这里的操作是将真实值的位置进行替换,替换成置信度
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 将填充的位置的值设置为0
        true_dist[:, self.padding_idx] = 0
        # 生成填充部分的掩码
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # 返回KL散度
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    # 计算损失和进行参数反向传播
    def __init__(self, generator, criterion, opt=None):
        """
        初始化
        :param generator: 生成器，transformer模块中的最后一层，这里将其单独拿出来而不直接放进transformer中的原因是：
            预测数据的是时候，我们需要利用之前的结果，但是我们只去最后一个作为本次输出，那么在进行输出时，只对最后一个进行输出，单独拿出来进行输出的线性变换，更灵活
        :param criterion: 标签平滑的类
        :param opt: 经wormup后的optimizer
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        """
        类做函数调用
        :param x: 经transformer解码后的结果
        :param y: 目标值
        :param norm: 本次数据有效的字符数，即，除去padding后的字符数
        :return:
        """
        # 进行输出
        x = self.generator(x)
        # 得到KL散度
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        # 反向椽笔
        loss.backward()
        if self.opt is not None:
            # 参数更新
            self.opt.step()
            # 优化器梯度置0
            self.opt.optimizer.zero_grad()
        # 返回损失
        return loss.data.item() * norm.float()


class NoamOpt:
    # warmup
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        初始化
        :param model_size: 词嵌入维度
        :param factor:
        :param warmup:
        :param optimizer:
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        # 学习率更新
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        # 学习率更新函数
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

class Parameters():
    def __init__(self):
        self.en_path=r''

