# 该模块是训练transformer的，所以主要的部分在于训练的数据集dataloader怎么设置以及如何进行epoch训练
'''
# Part1主要是引入一些库的函数
'''
import torch
from torch import nn
from transformer import Transformer
from dataset import de_vocab, en_vocab, de_preprocess, en_preprocess, train_dataset, PAD_IDX
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import SEQ_MAX_LEN

'''
# Part2 设计一个dataset数据集，继承于Dataset,dataset需要实现的功能
# 1. 在初始化就要初始化好训练集和测试集合。目前比较简单只有训练数据，所以初始化的时候只需要设计两个list，作为输入和输出就行。
# 2. 需要设计一些函数，返回数据集的一些信息，比如数据集的长度，等等

'''


class DeEnDataset(Dataset):
    def __init__(self):
        super().__init__()

        # 初始化的时候，要设置好所有的初始数据,这里的数据存储，主要是通过，list来存储的，并且输入和输出是分开存储的。
        self.enc_x = []
        self.dec_x = []
        for de, en in train_dataset:
            # 第一步分词,预处理
            de_tokens, de_ids = de_preprocess(de)
            en_tokens, en_ids = en_preprocess(en)
            # 判断序列长度是否超限度，对于超限度的句子直接去除了,这里的目的单纯是因为这种长序列的少见，以及内存，以及训练效果不佳啥的，实际是可以训练的
            if len(de_ids) > SEQ_MAX_LEN or len(en_ids) > SEQ_MAX_LEN:
                continue
            # 一个是decoder_x(输出的x)，一个是encoder_x(输入的x)
            self.enc_x.append(de_ids)
            self.dec_x.append(en_ids)


    # 获取长度
    def __len__(self):
        return len(self.enc_x)

    # 获取对应元素
    def __getitem__(self, index):
        return self.enc_x[index], self.dec_x[index]


'''
# Part3 设计一个collate_fn函数，用于Dataloader进行批量输出，包括tensor化和pad
'''


# 目前存在一个问题就是得到的数据是长度不一样的，可能需要进行PAD，所以得需要利用一个函数
# 对数据进行PAD，但是这个其实直接嵌入Dataset，感觉是最方便的，哈哈，直接得到每个元素等长的list。

# 但是Dataloader里面因为有cllate_fn选项，也就是获得一个batch的两个list之后对数据进行处理，然后进行批次输出
# 这里输入的是一个batch，格式是[(input,output),.....],也就是Dataloader里面会自动把两个list进行了zipped的，然后取一个batch。

def collate_fn(batch):
    enc_index_batch = []
    dec_index_batch = []

    # 遍历tensor化，到list里面去
    for enc_x, dec_x in batch:
        enc_index_batch.append(torch.tensor(enc_x, dtype=torch.long))
        dec_index_batch.append(torch.tensor(dec_x, dtype=torch.long))

    # 然后进行padding，因为pad_sequence只能在tensorlist用，应该batch_first表示张量以batch为第一个维度也就是(batch,seq_len)
    pad_enc_x = pad_sequence(enc_index_batch, batch_first=True, padding_value=PAD_IDX)
    pad_dec_x = pad_sequence(dec_index_batch, batch_first=True, padding_value=PAD_IDX)

    # 形状为 (batch, batchmax_seq_len),所以可能存在不同batch，这个句子长度是不一样的。
    # 所以为什么 position_emdding 里面有个 seq_max_len = 5000，然后取其前面部分的，因为每个batch可能句子长度不一样,所以位置编码要随时适应句子长度。
    return pad_enc_x, pad_dec_x


'''
# Part4 测试，真正开始训练
'''

if __name__=='__main__':
    dataset = DeEnDataset()
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True, collate_fn=collate_fn)

    # 尝试看看有没有现有模型，如果有现有模型就加载进行后训练，反之则创建一个进行重新训练，这里主要是用于前向传播
    try:
        transformer = torch.load('checkpoints/model.pth')
    except:
        transformer = Transformer(
            de_vocab_size=len(de_vocab),
            en_vocab_size=len(en_vocab),
            emd_size=512,
            head=8,
            q_k_size=64,
            v_size=64,
            f_size=2048,
            nums_encoder_block=6,
            nums_decoder_block=6,
            dropout=0.1,
            seq_max_len=SEQ_MAX_LEN
        )

    # 初始化损失
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 初始化优化器，反向传播更新参数用
    optimizer = torch.optim.SGD(transformer.parameters(), lr=1e-1, momentum=0.99)

    # 开始训练
    # 注意，对于module.train来说，如果模型中有Dropout以及Batch_norm等模块是一定要用的，因为训练的时候需要随机丢弃啥的，和测试的时候是不一样的。
    # 因为设计了.train()模型和.eval()模型，来区分模型不同的状态。
    transformer.train()

    EPOCHS = 300
    for epoch in range(EPOCHS):
        batch_i = 0
        loss_sum = 0
        for pad_enc_x, pad_dec_x in dataloader:
            # 这里一个去掉第一个词，一个去掉最后一个词，有说法的(所以相当于预测每下个词的概率)
            real_dec_z = pad_dec_x[:, 1:]  # decoder正确输出
            # encoder的输入
            pad_enc_x = pad_enc_x
            # decoder的输入
            pad_dec_x = pad_dec_x[:, :-1]  # decoder实际输入
            # 得到的对下个词的预测结果。
            dec_z = transformer(pad_enc_x, pad_dec_x)  # decoder实际输出

            batch_i += 1
            # 把(batch_size,seq_len-1,voab)平为(batch_size * seq_len-1,voab),从而和(batch_size * seq_len)进行计算损失。
            # 交叉上损失来说，计算的是(kind_num,vocab_size)和(kind_num)之间的差距，也就是真实数据只是个index，而预测出来是概率。
            loss = loss_fn(dec_z.reshape(-1, dec_z.size()[-1]), real_dec_z.reshape(-1))  # 把整个batch中的所有词拉平
            loss_sum += loss.item()
            print('epoch:{} batch:{} loss:{}'.format(epoch, batch_i, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(transformer, 'checkpoints/model.pth'.format(epoch))