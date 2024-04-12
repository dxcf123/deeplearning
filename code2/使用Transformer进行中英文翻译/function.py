import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

import classmodel


def subsequent_mask(size):
    """
    注意力机制掩码生成
    :param size: 句子长度
    :return: 注意力掩码
    """
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


def train():
    # 数据集
    en_path = r'H:\datasets\data\training-parallel-nc-v13\news-commentary-v13.zh-en.en'
    ch_path = r'H:\datasets\data\training-parallel-nc-v13\news-commentary-v13.zh-en.zh'
    # 加载tokenizer
    tokenizer = classmodel.Tokenizer(en_path, ch_path, count_min=50)
    # 训练设备
    device = 'cuda'
    # 加载模型
    model = classmodel.Transformer(tokenizer)
    # 模型初始化
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    # 将模型移动到训练设备上
    model = model.to(device)
    # 标签平滑与梯度更新
    criteria = classmodel.LabelSmoothing(tokenizer.get_ch_vocab_size(), tokenizer.ch_word_2_index['<pad>'])
    # warmup 动态学习率
    optimizer = classmodel.NoamOpt(256, 1, 2000,
                                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    # 损失计算
    lossF = classmodel.SimpleLossCompute(model.generator, criteria, optimizer)
    # 训练次数
    epochs = 10
    model.train()
    for epoch in range(epochs):
        lod = 0
        while True:
            # 获取dataloader，p=1时，数据集被用完一次
            data_loader, p = tokenizer.get_dataloader()
            if p:
                break
            # 遍历dataloader
            for index, data in enumerate(data_loader):
                # 获得源数据与目标数据
                src, tgt = data
                # 处理一个batch，获得掩码等相关内容
                batch = classmodel.Batch(src, tgt, tokenizer=tokenizer, device=device)
                # 训练
                out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
                # 计算损失
                out = lossF(out, batch.trg_y, batch.ntokens)
                # 清空显存中没用的内容
                torch.cuda.empty_cache()
                if index % 100 == 0:
                    model.eval()
                    print(epoch, lod, index, out / batch.ntokens)
                    # 模型预测
                    x = 'I have a dream!'
                    # y = ['']
                    # 编码
                    x = tokenizer.en_encode(x)
                    with torch.no_grad():
                        predict(x, model, tokenizer)
                    model.train()
            lod += 1
        torch.save(model.state_dict(), f'./model/translation_{epoch}.pt')


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           Variable(ys),
                           src_mask,
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, i])
        # print('prob', prob)
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        # print(next_word)
    return ys


def predict(data, model, tokenizer, device='cuda'):
    """
    在data上用训练好的模型进行预测，打印模型翻译结果
    """
    # 梯度清零
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for i in range(len(data)):
            # 打印待翻译的英文语句

            # 将当前以单词id表示的英文语句数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(np.array(data[i])).long().to(device)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != tokenizer.en_word_2_index['<pad>']).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=10, start_symbol=tokenizer.ch_word_2_index['<bos>'])
            # 初始化一个用于存放模型翻译结果语句单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = tokenizer.ch_index_2_word[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前语句的翻译结果列表
                if sym != '<eos>':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            # 打印模型翻译输出的中文语句结果
            print("translation: %s" % " ".join(translation))


def eval_model():
    en_path = r'H:\datasets\data\training-parallel-nc-v13\news-commentary-v13.zh-en.en'
    ch_path = r'H:\datasets\data\training-parallel-nc-v13\news-commentary-v13.zh-en.zh'
    tokenizer = classmodel.Tokenizer(en_path, ch_path, count_min=50)
    device = 'cuda'
    model = classmodel.Transformer(tokenizer)
    model.load_state_dict(torch.load('./model/translation_5.pt'))
    model = model.to(device)
    while True:
        x = input('输入需要翻译的句子: ')
        x = x if x != "" else 'I have a dream!'
        # x = 'I have a dream!'
        model.eval()
        x = tokenizer.en_encode(x)
        with torch.no_grad():
            predict(x, model, tokenizer)
