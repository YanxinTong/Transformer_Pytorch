# 该板块主要是对解码器进行串接，实现得到解码器部分
# 输入为x,还没嵌入的，但是PAD好的输入，输出需要对注意力值进行线性转化和softmax，最后得到一个单维向量，长度为词库大小。
'''
# Part1 导入库函数
'''
import torch
from torch import nn
from dataset import train_dataset, de_vocab, en_vocab, de_preprocess, en_preprocess,PAD_IDX
from encoder import Encoder
from decoder_block import DecoderBlock
from emb import EmbeddingWithPosition

'''
# Part2 设计解码器的类
'''


class Decoder(nn.Module):
    def __init__(self, en_vocab_size, emd_size, nums_decoder_block, head, q_k_size, v_size, f_size,seq_max_len=5000,dropout_rate=0.1):
        super().__init__()
        self.nums_decoder_block=nums_decoder_block
        # 首先对x进行编码
        self.emd = EmbeddingWithPosition(vocab_size=en_vocab_size, emd_size=emd_size,seq_max_len=seq_max_len,dropout_rate=dropout_rate)
        # 然后输入n个编码器
        self.decoder_list = nn.ModuleList()
        for _ in range(nums_decoder_block):
            self.decoder_list.append(
                DecoderBlock(head=head, emd_size=emd_size, q_k_size=q_k_size, v_size=v_size, f_size=f_size))
        # 然后需要线性化和softmax,目前是(batch_size,q_sqen_len,emd)
        # 得到(batch_size,vocab_size)
        self.linear1=nn.Linear(emd_size,en_vocab_size)
        self.softmax=nn.Softmax(-1)

    def forward(self, x, encoder_z,encoder_x): # encoder_x是编码器的输入(batch_size,q_seq_len)
        # x(batch_size,q_sqen_len)
        # 首先对解码器输入的padding位置进行掩码设置。
        mask1=(x==PAD_IDX).unsqueeze(1) # (batch_size,1,q_seq_len)
        mask1.expand(-1,x.size()[1],-1)  # (batch_size,q_seq_len,q_seq_len)
        # 然后要对解码器的输入的上半部分也取True然后和mask1或一下(也就是符号|),注意True表示需要隐藏的位置。
        # 注意：torch.tril 和 torch.triu 的区别就是决定矩阵的上半部分(不包含对角线)还是下半部分(不包含对角线)置为0,diagonal=1,表示置0的区域向上移动一行
        mask1=mask1 | torch.triu(torch.ones(mask1.size()[-1],mask1.size()[-1]),diagonal=1).bool().unsqueeze(0).expand(mask1.size()[0],-1,-1)


        # 然后对编码器的mask2进行掩码设置。在交叉注意力中，Padding 掩码的区域由K 和 V 的来源决定，
        # 而不是由Q 的来源决定。这确保了来自Q 的查询只关注K 中有效的信息位置。

        mask2 = (encoder_x == PAD_IDX).unsqueeze(1) # (batch_size,1,q_seq_len)
        mask2.expand(-1, encoder_x.size()[1], -1) # (batch_size,1,q_seq_len)

        x=self.emd(x)  # (batch_size,q_sqen_len,emd)

        # 进入解码器
        output=x
        for i in range(self.nums_decoder_block):
            output = self.decoder_list[i](output,encoder_z,mask1,mask2)

        # 输出进行线性层和softmax
        output=self.linear1(output)
        output=self.softmax(output)
        return output


if __name__ == '__main__':
    # 取2个de句子转词ID序列，输入给encoder
    de_tokens1, de_ids1 = de_preprocess(train_dataset[0][0])
    de_tokens2, de_ids2 = de_preprocess(train_dataset[1][0])
    # 对应2个en句子转词ID序列，再做embedding，输入给decoder
    en_tokens1, en_ids1 = en_preprocess(train_dataset[0][1])
    en_tokens2, en_ids2 = en_preprocess(train_dataset[1][1])

    # de句子组成batch并padding对齐
    if len(de_ids1) < len(de_ids2):
        de_ids1.extend([PAD_IDX] * (len(de_ids2) - len(de_ids1)))
    elif len(de_ids1) > len(de_ids2):
        de_ids2.extend([PAD_IDX] * (len(de_ids1) - len(de_ids2)))

    enc_x_batch = torch.tensor([de_ids1, de_ids2], dtype=torch.long)
    print('enc_x_batch batch:', enc_x_batch.size())

    # en句子组成batch并padding对齐
    if len(en_ids1) < len(en_ids2):
        en_ids1.extend([PAD_IDX] * (len(en_ids2) - len(en_ids1)))
    elif len(en_ids1) > len(en_ids2):
        en_ids2.extend([PAD_IDX] * (len(en_ids1) - len(en_ids2)))

    dec_x_batch = torch.tensor([en_ids1, en_ids2], dtype=torch.long)
    print('dec_x_batch batch:', dec_x_batch.size())

    # Encoder编码,输出每个词的编码向量
    enc = Encoder(vocab_size=len(de_vocab), emd_size=128, q_k_size=256, v_size=512, f_size=512, head=8, nums_encoderblock=3)
    enc_outputs = enc(enc_x_batch)
    print('encoder outputs:', enc_outputs.size())

    # Decoder编码,输出每个词对应下一个词的概率
    dec = Decoder(en_vocab_size=len(en_vocab), emd_size=128, q_k_size=256, v_size=512, f_size=512, head=8, nums_decoder_block=3)
    enc_outputs = dec(dec_x_batch, enc_outputs, enc_x_batch)
    print(enc_outputs)
    print('decoder outputs:', enc_outputs.size())


