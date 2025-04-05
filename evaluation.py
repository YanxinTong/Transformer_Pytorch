# 该模块主要是为了实现对句子进行翻译，主要分为两个部分，一个是先编码，然后是循环生成解码，直到生成了句子结束的符号。

'''
# Part1引入类
'''
import torch

from transformer import Transformer
from dataset import de_preprocess, PAD_IDX, EOS_IDX, BOS_IDX, UNK_IDX, en_vocab
from config import SEQ_MAX_LEN


def translate(transformer, de_seq):
    # de_seq 是德语句子, 首先进行预处理
    # 预处理完之后可能会存在未知的序列，哦哦哦，在preprocess里面已经处理过了
    de_tokens, de_indexs = de_preprocess(de_seq)

    # 对输入的德语句子要进行编码(1,seq_len,emdding)
    de_indexs = torch.Tensor(de_indexs, dtype=torch.long)
    encoder_z = transformer.encode(de_indexs)

    # decoder阶段

    # 添加kv_cache 的step1,开启和关闭kv_cache
    transformer.decoder.open_kvcache()

    # 生成输入的英语句子index
    en_indexs = [BOS_IDX]

    # 循环利用解码器进行生成
    while len(en_indexs) < SEQ_MAX_LEN:
        dec_indexs = torch.Tensor(en_indexs, dtype=torch.long)  # (batch,seq_len)
        decoder_z = transformer.decode(dec_indexs, encoder_z, de_indexs)  # (batch,seq_len,vocab_size)
        # 选取最后一个维度的预测概率
        next_tokens_p = decoder_z[0, dec_indexs.size(-1) - 1, :]
        # 获取最大的那个id
        next_token_id = torch.argmax(next_tokens_p)
        # 得到预测结果
        en_indexs.append(next_token_id)
        # 如果下一个词的预测结果为句子结尾就结束
        if next_token_id == EOS_IDX:
            break

    transformer.decoder.close_kv_cache()


    # 返回生成的结果,先去除无意义的index
    en_indexs = [id for id in en_indexs if id not in [PAD_IDX, EOS_IDX, BOS_IDX, UNK_IDX]]

    # 然后对其进行词库返回
    result=en_vocab.lookup_tokens(en_indexs)
    return ' '.join(result)

if __name__=='__main__':
    # 下载模型(pth文件可以是只有参数也可以是模型+参数，如果只有参数py文件需要增加网络结构，并且引入。)
    transformer=torch.load('checkpoints/model.pth')

    # 开启测试的模式
    transformer.eval()

    # 测试
    print(translate(transformer,'Zwei Männer unterhalten sich mit zwei Frauen'))
