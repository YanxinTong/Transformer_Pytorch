'''
该步实现，多头注意力机制，定义一个多头注意力机制的类：
输入: x为位置编码后的向量，维度为(batch_size,seq_len,emd_size)
输出：x为注意力值z,(batch_size,)
'''
'''
# Part1引入库函数
'''
import torch
from torch import nn
from dataset import de_vocab,de_preprocess,train_dataset
from emb import EmbeddingWithPosition
import math


'''
需要注意的一点，无论是交叉还是子注意力，通过公式我们可以知道，三者要相乘，所以进行线性转化后的KQV，应该满足
Q(q_seq_len,emd),K(k_v_seq_len,emd),V=(k_v_seq_len,emd_v)
也就是和WQ,WK,WV相乘之后得到的，Q和K的嵌入维度要相同，K和V的seq_len要相同。
因此，对于WK和WQ而言,因为只改变最后的一个维度因此，第二维度数值要相同。
'''
class MultiHeadAttention(nn.Module):
    def __init__(self,head,emd_size,q_k_size,v_size,dropout_rate=0.1):
        # 继承父类
        super().__init__()
        self.head=head
        # 我们需要定义一些模块，比如生成KQV矩阵
        # 因为要保留seq_len，所以把emd_size去掉,并且因为是多头把emd进行拆分，而这里拆分用的是Linear层，然后把头给拿出去
        # 主题q和k的size要一直相同，因为要q要和k进行矩阵乘法变成(seq_len,seq_len)
        self.Wk=nn.Linear(emd_size,q_k_size*head)
        self.Wq=nn.Linear(emd_size,q_k_size*head)
        self.Wv=nn.Linear(emd_size,v_size*head)
        self.dropout=nn.Dropout(dropout_rate)

        #  添加KV_cache 的step3 初始化记录的位置
        self.kv_cache_type = ''
        # 利用字典进行存储
        self.kv_cache = {}
    def set_kvcache (self,kv_cache_type=''):
        self.kv_cache_type=kv_cache_type
        # 利用字典进行存储
        self.kv_cache={}

    def forward(self,x_q,x_k_v,mask_pad):
        #  添加KV_cachestep4 对kv矩阵进行计算
        if self.kv_cache_type=='selfattn':
            x_q=x_q[:,-1,:] # (bat_size,1,emd_size)
            x_k_v=x_k_v[:,-1:]  # (bat_size,1,emd_size)
            q = self.Wq(x_q) # (bat_size,1,q_k_size*head)
            k = self.Wk(x_k_v) # (bat_size,1,q_k_size*head)
            v = self.Wv(x_k_v) # (bat_size,1,v_size*head)

            if 'Q' in self.kv_cache:
                q=torch.concat((self.kv_cache['Q'],q),dim=1) # (bat_size,seq_q_len,q_k_size*head)
            if 'K' in self.kv_cache:
                k=torch.concat((self.kv_cache['K'],k),dim=1) # (bat_size,seq_k_len,q_k_size*head)
            if 'V' in self.kv_cache:
                v=torch.concat((self.kv_cache['V'],v),dim=1) # (bat_size,seq_v_len,q_k_size*head)

            self.kv_cache.update({'Q':q.detach(),'K':k.detach(),'V':v.detach()})

        elif self.kv_cache_type=='crossattn':
            x_q = x_q[:, -1, :]
            q = self.Wq(x_q)

            if 'Q' in self.kv_cache:
                q = torch.concat((self.kv_cache['Q'], q), dim=1)
            if 'K' in self.kv_cache:
                k = self.kv_cache['K']
            else:
                k=self.Wk(x_k_v)
            if 'V' in self.kv_cache:
                v = self.kv_cache['V']
            else:
                v=v=self.Wv(x_k_v)

            self.kv_cache.update({'Q': q.detach(), 'K': k.detach(), 'V': v.detach()})

            self.kv_cache.update({'Q': q.detach(), 'K': k.detach(), 'V': v.detach()})
        else:# 训练阶段没开启kv_cache时候

            # 第一步分头,得先变成头的倍数再分头
            # 为了更加通用与交叉注意力，所以，把kv输入同源保证seq_len相同
            k=self.Wk(x_k_v)
            v=self.Wv(x_k_v)
            q=self.Wq(x_q)
        '''
        此时
        Q(batch,q_seq_len, emd), K(batch,k_v_seq_len, emd), V = (batch,k_v_seq_len, emd_v)
        然后先把头要进行拿出来，对于最后一个维度,拿到batch后面,但是为了不破坏维度，得一步步来，先把head从emd提取，然后利用transopose转换到前面
        '''
        q = q.reshape(q.size()[0], q.size()[1], self.head, -1).transpose(1, 2) # (batch, head, q_seq_len, q_k_size)
        k = k.reshape(k.size()[0], k.size()[1], self.head, -1).transpose(1, 2) # (batch, head, v_k_seq_len, q_k_size)
        v = v.reshape(v.size()[0], v.size()[1], self.head, -1).transpose(1, 2) # (batch, head, v_k_seq_len, v_size)
        # 为了q和k要相乘，所以需要对K进行转秩
        k=k.transpose(2,3) # (batch, head, q_k_size, v_k_seq_len)

        # 记得要除根号dk也就是单头的嵌入维度
        atten=torch.matmul(q,k)/math.sqrt(q.size()[-1]) # (batch,head,q_seq_len,v_k_seq_len)

        # 正常多头注意力，需要对padding位置的进行掩码处理。输入的mask是全0矩阵为(batch_size,q_seq_len,q_seq_len)
        # 插头的维度，并进行对头的维度进行expand
        mask_pad = mask_pad.unsqueeze(1).expand(-1,self.head,-1,-1)
        # 对 mask 为 True 的位置赋值为极小值 -1e9
        atten = atten.masked_fill(mask_pad, -1e9)

        # 然后要进行softmax，进行归一话，从而作为权重进行查询，主要是对于最后一个维度，也就是一个q，一个1，先不管掩码注意力
        atten=torch.softmax(atten,-1)

        # 和v相乘得到注意力结果z
        z=torch.matmul(atten,v) # (batch, head, q_seq_len, v_size)

        # 然后要开始转回原来的样子
        z=z.transpose(1,2) # 第一步交换 (batch, q_seq_len, head, v_size)
        z=z.reshape(z.size()[0],z.size()[1],-1) # 第一步交换 (batch, q_seq_len, head * v_size)
        return z
if __name__=='__main__':
    # 需要一个batch的数据
    deyu=train_dataset[0][0]
    _,deyu_pro=de_preprocess(deyu)
    deyu_pro_tensor=torch.tensor(deyu_pro,dtype=torch.long)

    deyu_pro_tensor=deyu_pro_tensor.unsqueeze(0)
    # 对其进行编码
    emd=EmbeddingWithPosition(len(de_vocab),128)
    deyu_pro_tensor_encode=emd(deyu_pro_tensor)

    '''
    存在在一个小问题，就是到现在为止，还没开始做补全句子。
    '''
    # 输出多头注意力结果
    ma=MultiHeadAttention(head=8,emd_size=128,q_k_size=256,v_size=512)

    mask_pad=torch.zeros(1,deyu_pro_tensor.size()[1],deyu_pro_tensor.size()[1])

    print(ma(x_q=deyu_pro_tensor_encode,x_k_v=deyu_pro_tensor_encode,mask_pad=mask_pad))


















