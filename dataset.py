'''
# Part1 手撕transformer下载数据集
'''

# 第一步引入库函数,都是torchtext的子库
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k,Multi30k

'''
# Part2 下载数据集,通过网址下载：
'''
# 为 Multi30k 数据集指定训练集的下载链接，两个都是压缩包
multi30k.URL['train']="https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
# 为 Multi30k 数据集指定验证集的下载链接
multi30k.URL['valid']="https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
# 从训练集中获取，德语和英语的语言配对数据，数据类型为文本元组对的list
train_dataset=list(Multi30k(split='train',language_pair=('de','en')))

# print(train_dataset)
'''
# Part3 创建分词器，都是Spacy的分词器，后面是指定的预训练的不同语言的分词器
由于spacy分词下的有点烦，直接用个基础的分词器
'''

# # 作用：把词一个一个分开。
# # 英语的分词器
# de_tokenizer=get_tokenizer('spacy', language='de_core_news_sm')
# # 德语的分词器
# en_tokenizer=get_tokenizer('spacy', language='en_core_web_sm')

# 使用TorchText自带的basic_english分词器
en_tokenizer = get_tokenizer('basic_english')
de_tokenizer = get_tokenizer('basic_english')  # 如果适用德语


'''
# Part4 不同特殊词的类型
    UNK_IDX: 未知词（<unk>）的索引值，表示词表中不存在的单词，unknown
    PAD_IDX: 填充符（<pad>）的索引值，用于对齐序列长度,padding,用于填充，把所有的句子填充成一样长，便于按批处理。
    BOS_IDX: 序列开始符（<bos>）的索引值，标志序列的起始，begin of sentence
    EOS_IDX: 序列结束符（<eos>）的索引值，标志序列的结束, end of sentence
'''
# 定义特殊 token 的索引值
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# 定义特殊 token 的符号
UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM = '<unk>', '<pad>', '<bos>', '<eos>'

'''
# Part5：获取每个句子的tokens列表，便于建立词库
'''
# 初始化
de_tokens=[] # 德语token列表
en_tokens=[] # 英语token列表

# 遍历填入,分词后的句子。
for de,en in train_dataset:
    de_tokens.append(de_tokenizer(de))
    en_tokens.append(en_tokenizer(en))

'''
# Part6 对获得的词进行词库的建立。
对得到的句子tokens_list，形成词库，主要作用类似两个字典，分别是{index:词}和{词:index},或者说双向字典，用于查询
'''
# 德语词库的建立：其中specials的意思是特殊的词也加入词表，special_first的的意思是把这些词放在词表的开头，也就是，0，1，2，3
de_vocab=build_vocab_from_iterator(de_tokens,specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM],special_first=True) # 德语token词表
# 这里是默认把查不到的词，索引到unkonw的index里面。
de_vocab.set_default_index(UNK_IDX)

# 同理英语词库的建立
en_vocab=build_vocab_from_iterator(en_tokens,specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM],special_first=True) # 英语token词表
en_vocab.set_default_index(UNK_IDX)

'''
# Part7 对句子的特征进行预处理，主要是分词+前后添加句子开始和结束的符号
输入句子，原始句子或者分词后的句子都可以
返回：tokens_list(添加了前后开始结束符号)和index_list
'''

# 句子特征预处理
def de_preprocess(de_sentence):
    # 德语分词
    tokens=de_tokenizer(de_sentence)
    # 添加开始和结束符
    tokens=[BOS_SYM]+tokens+[EOS_SYM]
    # 查字典得到句子的tokens的index_list
    ids=de_vocab(tokens)
    # 返回句子分词结果和index_list结果
    return tokens,ids

def en_preprocess(en_sentence):
    tokens=en_tokenizer(en_sentence)
    tokens=[BOS_SYM]+tokens+[EOS_SYM]
    ids=en_vocab(tokens)
    return tokens,ids


if __name__ == '__main__':
    # 返回词表大小
    print('de vocab:', len(de_vocab))
    print('en vocab:', len(en_vocab))

    # 特征预处理
    de_sentence,en_sentence=train_dataset[0]

    # *号是解包的意思，把里面的元素拿出来
    print('de preprocess:',*de_preprocess(de_sentence))
    print('en preprocess:',*en_preprocess(en_sentence))

