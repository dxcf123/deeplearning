import sacrebleu


def bleu_score(hypothesis, reference):
    """
    计算两个句子之间的BLEU得分
    :param hypothesis: 预测的句子
    :param reference: 真实的目标句子
    :return: BLEU得分
    """
    score = [sacrebleu.corpus_bleu(hypothesis[i], reference[i]).score for i in range(len(hypothesis))]
    return score
