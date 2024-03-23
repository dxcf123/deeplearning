import os
import time
from torch import optim
from torch.utils.data import DataLoader
import classmodel
import nltk
import jieba
import torch
import pandas as pd


def read_data_split(path, path2):
    """
    读取文件
    :param path: 文件存放路径
    :param path2: 中间文件存放路径
    :return: 返回切分后的数据
    """
    if not os.path.exists(os.path.join(path2, 'en_data_split.txt')):  # 如果文件未经分词，则分词
        data = pd.read_csv(path, encoding='utf-8')
        en_data = list(data['english'])
        ch_data = list(data['chinese'])

        en_data_split = []  # 英文数据分词
        for i in en_data:
            en_data_split.append(nltk.word_tokenize(i))

        ch_data_split = []  # 中文数据分词
        for i in ch_data:
            ch_data_split.append(list(jieba.cut(i)))

        # 将分词后的数据写入文件
        with open(os.path.join(path2, 'en_data_split.txt'), 'w', encoding='utf-8') as f:
            f.write(str(en_data_split))
        with open(os.path.join(path2, 'ch_data_split.txt'), 'w', encoding='utf-8') as f:
            f.write(str(ch_data_split))
        return en_data_split, ch_data_split
    # 若经分词，则直接读取
    else:
        with open(os.path.join(path2, 'en_data_split.txt'), 'r', encoding='utf-8') as f:
            en_data_split = eval(f.read())
        with open(os.path.join(path2, 'ch_data_split.txt'), 'r', encoding='utf-8') as f:
            ch_data_split = eval(f.read())
        return en_data_split, ch_data_split


def pre_process1(path, path2):
    """
    对数据进行预处理，按英文句子长度进行排序，排序的作用是，后期训练时减少填充符的干扰
    :param path: 文件存放路径
    :param path2: 中间文件存放路径
    :return: 返回排序后的数据
    """
    # 判断是否已经有排序后的数据
    if not os.path.exists(os.path.join(path2, 'en_data_sort.txt')):
        # 进行数据分词
        en_data_split, ch_data_split = read_data_split(path, path2)
        en_len = []  # 添加每个句子的长度与索引，以元组形式(长度，索引)
        for i in range(len(en_data_split)):  # 遍历英文数据
            en_len.append((len(en_data_split[i]), i))  # 将长度与索引传入数组

        # 对列表进行排序 正序
        en_len.sort(key=lambda x: x[0], reverse=False)

        # 提取排序后的索引
        en_len_sort = [i[1] for i in en_len[5:]]  # 不要前5个数据的原因是它的翻译是错误的，其它句子是否有错误翻译，未知

        # 根据索引对数据进行排序
        en_data_sort = [en_data_split[i] for i in en_len_sort]
        ch_data_sort = [ch_data_split[i] for i in en_len_sort]

        with open(os.path.join(path2, 'en_data_sort.txt'), 'w', encoding='utf-8') as f:
            f.write(str(en_data_sort))
        with open(os.path.join(path2, 'ch_data_sort.txt'), 'w', encoding='utf-8') as f:
            f.write(str(ch_data_sort))
        # 返回排序后的数据
        return en_data_sort, ch_data_sort
    # 若经排序，则直接读取
    else:
        with open(os.path.join(path2, 'en_data_sort.txt'), 'r', encoding='utf-8') as f:
            en_data_sort = eval(f.read())
        with open(os.path.join(path2, 'ch_data_sort.txt'), 'r', encoding='utf-8') as f:
            ch_data_sort = eval(f.read())

        return en_data_sort, ch_data_sort


def pre_process2(path, path2):
    """
    统计字符数并填充字符
    这里会产生四份文件，分别为中英文到数字的索引与数字到中英文的索引
    :param path: 文件存放路径
    :param path2: 中间文件存放路径
    :return: 中英文到数字的索引与数字到中英文的索引
    """
    # 判断字符是否已经被统计
    if not os.path.exists(os.path.join(path2, 'en_word_2_index.txt.txt')):
        # 获取排序后的数据
        en_data_sort, ch_data_sort = pre_process1(path, path2)

        # 统计英文字符
        en_index_2_word = []  # 存储索引和对应单词
        en_word_2_index = dict()  # 存储单词和对应索引
        ch_index_2_word = []  # 存储索引和对应中文
        ch_word_2_index = dict()  # 存储中文和对应索引

        # 统计英文单词
        count = 0
        for i in en_data_sort:
            for j in i:
                if j not in en_word_2_index:
                    en_word_2_index[j] = count
                    en_index_2_word.append(j)
                    count += 1

        # 统计中文
        count = 0
        for i in ch_data_sort:
            for j in i:
                if j not in ch_word_2_index:
                    ch_word_2_index[j] = count
                    ch_index_2_word.append(j)
                    count += 1

        # 获取统计字符后的总字符数
        l1 = len(ch_index_2_word)
        l2 = len(en_index_2_word)

        # 添加填充字符
        # 英文添加'<pad>'，中文添加'<pad>'，'<bos>','<eos>',用于训练时判断句子的的开始与结束
        ch_word_2_index.update({'<pad>': l1, '<bos>': l1 + 1, '<eos>': l1 + 2})
        ch_index_2_word.extend(['<pad>', '<bos>', '<eos>'])

        en_word_2_index.update({'<pad>': l2})
        en_index_2_word.append('<pad>')

        # 将四组数据写入文件
        with open(os.path.join(path2, 'en_word_2_index.txt'), 'w', encoding='utf-8') as f:
            f.write(str(en_word_2_index))
        with open(os.path.join(path2, 'en_index_2_word.txt'), 'w', encoding='utf-8') as f:
            f.write(str(en_index_2_word))
        with open(os.path.join(path2, 'ch_word_2_index.txt'), 'w', encoding='utf-8') as f:
            f.write(str(ch_word_2_index))
        with open(os.path.join(path2, 'ch_index_2_word.txt'), 'w', encoding='utf-8') as f:
            f.write(str(ch_index_2_word))
        return en_word_2_index, en_index_2_word, ch_word_2_index, ch_index_2_word
    # 读取已有则直接读取
    else:
        with open(os.path.join(path2, 'en_word_2_index.txt'), 'r', encoding='utf-8') as f:
            en_word_2_index = eval(f.read())
        with open(os.path.join(path2, 'en_index_2_word.txt'), 'r', encoding='utf-8') as f:
            en_index_2_word = eval(f.read())
        with open(os.path.join(path2, 'ch_word_2_index.txt'), 'r', encoding='utf-8') as f:
            ch_word_2_index = eval(f.read())
        with open(os.path.join(path2, 'ch_index_2_word.txt'), 'r', encoding='utf-8') as f:
            ch_index_2_word = eval(f.read())
        return en_word_2_index, en_index_2_word, ch_word_2_index, ch_index_2_word


def read_data_ori(path, path2, start=0, end=None):
    """
    读取数据，用于获取数据集
    :param path: 原始数据集路径
    :param path2: 中间数据路径
    :param start: 数据截取开始的索引
    :param end: 数据截取结束的索引
    :return:
    """
    # 获取分词后且排过序的数据
    en_data, ch_data = pre_process1(path, path2)
    en_data = en_data[start:end]  # 选择需要的数据
    ch_data = ch_data[start:end]
    return en_data, ch_data


def your_translater(model, en_word_2_index, ch_word_2_index, ch_index_2_word, device):
    """
    使用模型进行翻译
    :param model: 模型
    :param en_word_2_index: 英文词到索引
    :param ch_word_2_index: 中文词到索引
    :param ch_index_2_word: 索引到中文词
    :param device: 设备
    :return:
    """
    while True:
        en_word = input("请输入您要翻译的英文句子：")  # 输入需要翻译的句子
        en_word = nltk.word_tokenize(en_word)
        # 将句子转换成索引且转换为tensor
        en_index = torch.tensor([[en_word_2_index[word] for word in en_word]], device=device)
        # 将英文句子输入模型的Encoder获得h0,c0
        h0, c0 = model.encoder(en_index)
        #  中文的开始是<bos>，因此将bos传入作为第一个中文输入，这里将<bos>对应的索引转换为tensor
        decoder_input = torch.tensor([[ch_word_2_index['<bos>']]], device=device)

        # 这里保存预测的结果
        result = []
        while True:  # 执行死循环进行预测
            # 传入decoder获得预测，每次输入一个字只能获取一个输出，因此才用死循环来预测，这里输出的h0,c0用于下一层循环的输入
            output, h0, c0 = model.decoder(decoder_input, h0, c0)
            # 将输出放入线性层转换纬度
            pre = model.linear(output)
            # 获取最大概率的索引
            pre = pre.argmax(dim=-1)
            print(pre)
            # 将索引转换为对应的字
            wor = ch_index_2_word[pre]
            print(wor)
            # 将结果保存起来
            result.append(wor)
            #  如果预测到<eos>或者结果超过300个字，则结束循环
            if wor == '<eos>' or len(result) > 50:
                break
            # 将这层的预测结果作为下一层的输入
            decoder_input = torch.tensor([[pre]], device=device)


def train(path, path2, e_embedding_dim=128, hidden_num=512, d_embedding_num=128, batchsize=5, epochs=10, device='cpu',
          lr=0.001):
    """
    模型训练
    :param path: 原数据文件夹
    :param path2:  中间数据文件夹
    :param e_embedding_dim : 英文嵌入维度
    :param hidden_num: 隐藏层维度
    :param d_embedding_num: 中文词嵌入纬度
    :param batchsize: 批量
    :param epochs: 迭代次数
    :param device: 训练设备
    :param lr: 学习率
    :return:模型
    """
    # 读取中英文对应的索引与索引对应的中英文
    en_word_2_index, en_index_2_word, ch_word_2_index, ch_index_2_word = pre_process2(path, path2)

    # 获取中英文的索引长度
    en_len = len(en_index_2_word)
    ch_len = len(ch_index_2_word)

    # 读取原始数据
    en_data, ch_data = read_data_ori(path, path2, start=0, end=26000)
    # en_data, ch_data = per_process_related_to_batchsize(en_data, ch_data, batchsize, ch_word_2_index, en_word_2_index)

    # 创建数据集
    dataset = classmodel.MyDataset(ch_data, en_data, ch_word_2_index, en_word_2_index, device)
    # 创建数据加载器  collate_fn用于将一个batch的数据流入到another_process处理
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, collate_fn=dataset.another_process)

    # 创建模型
    model = classmodel.Seq2Seq(e_embedding_dim, hidden_num, en_len, d_embedding_num, ch_len)

    # 将模型加载到device上
    model = model.to(device)

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    t1 = time.time()
    for e in range(epochs):
        for i, (en_index, ch_index) in enumerate(dataloader):
            loss = model(en_index, ch_index)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 1000 == 0:
                t2 = time.time()
                print(e, i, loss)
                print(t2 - t1)
                t1 = t2

    return model
