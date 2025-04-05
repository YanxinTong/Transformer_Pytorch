# 该模块主要用于实现词嵌入，然后加上位置向量,也就是把每次词嵌入为每个向量，然后对其加上位置编码。
# 当然前提是先把词list转为index。

'''
# Part1: 进行一些库的引入
'''
from torch import nn
import torch
# 我们的的dataset.py文件，来引入德语词表，德语的预处理函数，以及训练的数据集
from dataset import de_vocab,de_preprocess,train_dataset
import math

'''
 Part2: 实现嵌入以及位置编码，作为一个类
'''
class EmbeddingWithPosition(nn.Module):
    # 初始化参数，注意这里的seq_max_len是最长的位置序列，我们可能只会用到其前面的数据，具体得看输入的x的嵌入大小
    def __init__(self,vocab_size,emd_size,seq_max_len=5000,dropout_rate=0.1):
        # 继承父类
        super().__init__()
        # 初始化函数，用于index转化为向量(embdding)，这里决定要输入嵌入的向量总个数(即字典库的个数)，以及每个index需要嵌入的维度。
        # 输入(batch_size, sequence_length)，输出(batch_size, sequence_length, embedding_dim)。
        # 注意：参数可训练
        self.seq_emd=nn.Embedding(num_embeddings=vocab_size,embedding_dim= emd_size)

        # 位置编码，为一个句子中的每个位置也进行编码，一个位置的编码维度也为嵌入的维度，从而可以直接相加
        # 这里我们得知道一个句子统一的长度是多少(seq_max_len)，从而便于对所有可能的位置编码,unsqueeze表示在某个地方添加一个维度,这里是为了便于嵌入，所以需要额外的维度
        position_idx=torch.arange(0,seq_max_len,dtype=torch.float).unsqueeze(-1) # (seq_max_len,1)

        '''
        位置编码，为f(index/pow(10000,(2*i/emd_size))),f当index=2*i则为sin,反之为cos。
        由于频率 10000^{2i/d} 可能是一个非常极端的数，会导致计算不稳定。因此实际应用时，为避免数值上溢/下溢问题.
        先使用对数函数 log 将大范围的值(如10000)缩放到更合适的范围，使频率变化平滑。再使用 torch.exp(...)计算指数，恢复频率的实际值
        因此变为，f(index*exp(log(1/pow(10000,(2*i/emd_size)))) = f(index*exp(-log(10000)*(2i/emd_size))
        '''

        # 先计算函数内部的数据
        position_emd_fill=position_idx*torch.exp(-torch.arange(0,emd_size,2)/emd_size*math.log(10000.0))
        position_encoding=torch.zeros(seq_max_len,emd_size)
        # 对于嵌入的为偶数的f为sin
        position_encoding[:,0::2]=torch.sin(position_emd_fill)
        # 奇数为cos
        position_encoding[:,1::2]=torch.cos(position_emd_fill)

        self.register_buffer('position_encoding',position_encoding)

        # 用于防止过拟合的Dropout
        self.dropout=nn.Dropout(dropout_rate)
    # 前向传播
    def forward(self,x):
        # x(batch,seq_max_len)为输入,输出编码后的数据
        x_emd=self.seq_emd(x) # (batch,seq_len,emd_size)
        # 因为position_encoding的第二维是max_seq_len,所以我们只需要取其前面的作为位置嵌入编码
        x_emd_pos=x_emd+self.position_encoding.unsqueeze(0)[:,:x_emd.size()[1],:]
        return self.dropout(x_emd_pos)

if __name__ == '__main__':
    # 初始化类编码
    emd=EmbeddingWithPosition(len(de_vocab),128)
    # 取一个德语的句子
    de_yu=train_dataset[0][0]
    # 对德语进行预处理，并得到index
    _,de_yu_pro=de_preprocess(de_yu)
    print(len(de_yu_pro))
    # 对预处理的语言进行转化为tensor
    de_yu_pro_tr=torch.tensor(de_yu_pro,dtype=torch.long)
    # 将其进行转化为适合的维度，添加第一维度
    de_yu_pro_tr=emd(de_yu_pro_tr.unsqueeze(0))

    print(de_yu+'的编码为：\n',de_yu_pro_tr)



















