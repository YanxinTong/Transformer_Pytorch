import torch
# 获取基础参数信息
# 如果要指定第几块'cuda:1',指的是第二块
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 最长序列（受限于postition emb）
SEQ_MAX_LEN=5000