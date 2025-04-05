# 这个是解码器的block，和编码器来说多了一个掩码注意力机制，但是其实就是把掩码换一下即可，同时还对于第二个多头注意力机制的k_v和q不同源了
# 主要构成要素，输入嵌入好的句子，经过1.掩码注意力机制+残差归一化 2. 交叉注意力+残差归一化 3. 前向+残差归一化。保证输入输出同纬度(batch_size,seq_len,emding)
'''
# Part1 引入库函数
'''
import torch
from torch import nn
from multihead_attn import MultiHeadAttention
# 应该是用于测试
from dataset import train_dataset,de_preprocess,de_vocab,en_preprocess,en_vocab,PAD_IDX
from emb import EmbeddingWithPosition
from encoder import Encoder

'''
# Part2 写个类，实现EncoderBlock
'''
class DecoderBlock(nn.Module):
    def __init__(self,head,emd_size,q_k_size,v_size,f_size):
        super().__init__()
        # 首先要进行掩码多头注意力机制
        self.mask_multi_atten=MultiHeadAttention(head=head,emd_size=emd_size,q_k_size=q_k_size,v_size=v_size)
        self.linear1=nn.Linear(head*v_size,emd_size)
        # 归一化(填写的是最后一个的那个维度大小)
        self.norm1=nn.LayerNorm(emd_size)

        # 交叉注意力机制
        self.cross_multi_atten=MultiHeadAttention(head=head,emd_size=emd_size,q_k_size=q_k_size,v_size=v_size)
        self.linear2 = nn.Linear(head * v_size, emd_size)
        # 归一化(填写的是最后一个的那个维度大小)
        self.norm2 = nn.LayerNorm(emd_size)

        # 前向
        self.feedforward=nn.Sequential(
            nn.Linear(emd_size,f_size),
            nn.ReLU(),
            nn.Linear(f_size, emd_size)
        )
        self.norm3 = nn.LayerNorm(emd_size)

    #  增加KVcache的step2： 在Decoder里面添加开启和关闭kvcache。
    def open_kvcache(self):
        self.mask_multi_atten.set_kvcache(kv_cache_type='selfattn')
        self.cross_multi_atten.set_kvcache(kv_cache_type='crossattn')

    def close_kvcache(self):
        self.mask_multi_atten.set_kvcache(kv_cache_type='')
        self.cross_multi_atten.set_kvcache(kv_cache_type='')

    def forward(self, x, encoder_z, mask_1, mask_2): # x(batch_size,q_seq_len,emd_size)
        # 掩码注意力机制
        z1=self.mask_multi_atten(x_q=x, x_k_v=x, mask_pad=mask_1) # (batch_size,q_seq_len,head*v_size)
        z1=self.linear1(z1) # (batch_size,q_seq_len,emd_size)
        # 第一个残差归一化，得到第一层的输出output
        outpu1=self.norm1(z1+x) # (batch_size,q_seq_len,emd_size)

        # 交叉注意力机制，把output作为q，编码器作为k_v
        z2=self.cross_multi_atten(x_q=outpu1, x_k_v=encoder_z, mask_pad=mask_2) # (batch_size,q_seq_len,head*v_size)
        # 第二个残差归一化
        z2 = self.linear1(z2) # (batch_size,q_seq_len,emd_size)
        output2=self.norm2(z2+outpu1) # (batch_size,q_seq_len,emd_size)

        # 前向
        z3=self.feedforward(output2) # (batch_size,q_seq_len,emd_size)
        # 第三个残差归一化
        output3 = self.norm3(z3 + output2) # (batch_size,q_seq_len,emd_size)
        return output3


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

    # 生成decoder所需的掩码
    first_attn_mask = (dec_x_batch == PAD_IDX).unsqueeze(1).expand(dec_x_batch.size()[0], dec_x_batch.size()[1],
                                                                   dec_x_batch.size()[1])  # 目标序列的pad掩码
    first_attn_mask = first_attn_mask | torch.triu(torch.ones(dec_x_batch.size()[1], dec_x_batch.size()[1]),
                                                   diagonal=1).bool().unsqueeze(0).expand(dec_x_batch.size()[0], -1,
                                                                                          -1) # &目标序列的向后看掩码
    print('first_attn_mask:', first_attn_mask.size())
    # 根据来源序列的pad掩码，遮盖decoder每个Q对encoder输出K的注意力
    second_attn_mask = (enc_x_batch == PAD_IDX).unsqueeze(1).expand(enc_x_batch.size()[0], dec_x_batch.size()[1],
                                                                    enc_x_batch.size()[
                                                                        1])  # (batch_size,target_len,src_len)
    print('second_attn_mask:', second_attn_mask.size())

    first_attn_mask = first_attn_mask
    second_attn_mask = second_attn_mask

    # Decoder输入做emb先
    emb = EmbeddingWithPosition(len(en_vocab), 128)
    dec_x_emb_batch = emb(dec_x_batch)
    print('dec_x_emb_batch:', dec_x_emb_batch.size())

    # 5个Decoder block堆叠
    decoder_blocks = []
    for i in range(5):
        decoder_blocks.append(DecoderBlock(emd_size=128, q_k_size=256, v_size=512, f_size=512, head=8))

    for i in range(5):
        dec_x_emb_batch = decoder_blocks[i](dec_x_emb_batch, enc_outputs, first_attn_mask, second_attn_mask)
    print('decoder_outputs:', dec_x_emb_batch.size())



