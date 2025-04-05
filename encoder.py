# encoder,主要组成部分是输入的嵌入，encoder_block的组成，最后得到多个编码器组合的输出z

'''
# Part1 进行库的导入

'''
import torch
from torch import nn
from emb import EmbeddingWithPosition
from dataset import de_vocab,train_dataset,de_preprocess,PAD_IDX
from encoder_block import EncoderBlock


'''
# Part2 定义编码器的这个类
'''
class Encoder(nn.Module):
    def __init__(self,vocab_size,emd_size,head,q_k_size,v_size,f_size,nums_encoderblock=5,seq_max_len=5000,dropout_rate=0.1):
        super().__init__()
        self.nums_encoderblock=nums_encoderblock
        # 定义编码器
        self.emd=EmbeddingWithPosition(vocab_size=vocab_size,emd_size=emd_size,seq_max_len=seq_max_len,dropout_rate=dropout_rate)
        # encoder block
        self.encoder_block_list=[]
        for i in range(nums_encoderblock):
            self.encoder_block_list.append(EncoderBlock(emd_size=emd_size,f_size=f_size,head=head,v_size=v_size,q_k_size=q_k_size))



    def forward(self,x): #输出是原始的没编码过的list，不定长，(batch_size,q_seq_len)都形成不了矩阵？
        # 前提此时的x已经是PAD过的矩阵。为(batch_size,q_seq_len)
        mask_pad=(x==PAD_IDX).unsqueeze(1) # (batch_size,1,q_seq_len)
        mask_pad=mask_pad.expand(-1,mask_pad.size()[1],-1) # (batch_size,q_seq_len,q_seq_len)
        mask_pad=mask_pad
        # 进行编码
        x = self.emd(x) # (batch_size,seq_len,emd_size)
        output=x # (batch_size,seq_len,emd_size)
        for i in range(self.nums_encoderblock):
            output=self.encoder_block_list[i](output,mask_pad) # (batch_size,seq_len,emd_size)
        return output


if __name__ == '__main__':
    # 取2个de句子转词ID序列
    de_tokens1, de_ids1 = de_preprocess(train_dataset[0][0])
    de_tokens2, de_ids2 = de_preprocess(train_dataset[1][0])

    # 组成batch并padding对齐
    if len(de_ids1) < len(de_ids2):
        de_ids1.extend([PAD_IDX] * (len(de_ids2) - len(de_ids1)))
    elif len(de_ids1) > len(de_ids2):
        de_ids2.extend([PAD_IDX] * (len(de_ids1) - len(de_ids2)))

    batch = torch.tensor([de_ids1, de_ids2], dtype=torch.long)
    print('batch:', batch.size())

    # Encoder编码
    encoder = Encoder(vocab_size=len(de_vocab), emd_size=128, q_k_size=256, v_size=512, f_size=512, head=8,
                      nums_encoderblock=3)
    z = encoder.forward(batch)
    print('encoder outputs:', z.size())







