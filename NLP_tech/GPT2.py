import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# input embedding
# 将输入序列转化为向量表示
class GPT2Embedding(nn.Module):
    def __init__(self,vocab_size,embed_dim,max_seq_len):
        super(GPT2Embedding,self).__init__()
        self.token_embedding = nn.Embedding(vocab_size,embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len,embed_dim)
        self.embed_dim = embed_dim

    def forward(self,x):
        # 对输入的序列进行嵌入处理
        # 参数x是输入序列,形状是(batch_size,seq_len)
        # 这里的 seq_len 是序列的长度，即输入序列中每个样本的单词数
        # 假设 x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 形状为 (batch_size=2, seq_len=3)
        # seq_len = x.size(1)  # seq_len = 3
        # x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        # print(x.size())  # 输出 torch.Size([2, 3])
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0) # 创建序列的位置信息
        token_embed = self.token_embedding(x)   #词嵌入
        position_embed = self.position_embedding(positions) #位置编码
        return token_embed + position_embed


def scaled_dot_product_attention(Q,K,V,mask=None):
    """
    实现Scaled Dot-Product Attention
    :param Q: 查询向量,形状(batch_size,num_heads, seq_len,d_K)
    :param K: 键向量，形状 (batch_size, num_heads, seq_len, d_k)
    :param V: 值向量，形状 (batch_size, num_heads, seq_len, d_v)
    :param mask: 注意力掩码，形状 (batch_size, 1, seq_len, seq_len)，可选
    :return: 注意力输出，形状 (batch_size, num_heads, seq_len, d_v)
    """
    # dot product ( calculate the score of Q and K )
    score = torch.matmul(Q,K.transpose(-2,-1))  # 形状: (batch_size, num_heads, seq_len, seq_len)

    # 缩放分数
    d_k = Q.size(-1)
    scaled_scores = score / torch.sqrt(torch.tensor(d_k,dtype=torch.float32)) # 缩放

    # if apply mask then use
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask==0, float('-inf')) #屏蔽无效位置

    # calculate the attention weight(softmax)
    attention_weights = F.softmax(scaled_scores,dim=-1) # 形状: (batch_size, num_heads, seq_len, seq_len)

    # 用注意力权重加权 V
    output = torch.matmul(attention_weights,V)
    return output,attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim, num_heads):
        super(MultiHeadAttention,self).__init__()
        self.embed_dim = embed_dim  # 输入嵌入的维度
        self.num_heads = num_heads  # 注意力头的数量
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        assert embed_dim % num_heads == 0, "embed_dim must be divided by num_heads"

        # define the linear layer of Q,K,V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # define W_o 线性变换矩阵
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self,Q,K,V,mask = None):
        batch_size = Q.size(0)

        # linear transform , generates multi-head Q,K,V
        Q = self.W_q(Q)  # (batch_size, seq_len, embed_dim)
        K = self.W_k(K)  # (batch_size, seq_len, embed_dim)
        V = self.W_v(V)  # (batch_size, seq_len, embed_dim)

        # divide into multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # 形状变为 (batch_size, num_heads, seq_len, head_dim)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q,K.transpose(-2,-1))
        # scaled
        scores = scores / torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores,dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # concat
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # 形状变换 (batch_size, seq_len, embed_dim)
        output = self.W_o(attention_output)

        return output,attention_weights

class AddNorm(nn.Module):
    def __init__(self,embed_dim, eps=1e-6):
        super(AddNorm,self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim,eps)

    def forward(self,x,sublayer_output):
        """
        :param x: 原始输入张量 (batch_size, seq_len, embed_dim)
        :param sublayer_output: 子层输出张量 (batch_size, seq_len, embed_dim)
        :return: 残差连接和归一化后的张量
        """
        return self.layer_norm(x + sublayer_output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim,activation=nn.ReLU):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = activation()

    def forward(self, x):
        """
        :param x: 输入张量 (batch_size, seq_len, embed_dim)
        :return: 经过前馈网络的输出
        """
        x = self.activation(self.fc1(x))  # 第1层
        return self.fc2(x)  # 第2层


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.add_norm1 = AddNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim)
        self.add_norm2 = AddNorm(embed_dim)

    def forward(self, x, mask=None):
        # Step 1: Multi-Head Attention + Add & Norm
        attn_output, _ = self.attention(x, x, x, mask)  # 自注意力
        x = self.add_norm1(x, attn_output)  # 残差连接 + 归一化

        # Step 2: Feed Forward Network + Add & Norm
        ffn_output = self.ffn(x)  # 前馈网络
        x = self.add_norm2(x, ffn_output)  # 残差连接 + 归一化

        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len):
        super(GPT2, self).__init__()
        self.embedding = GPT2Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)  # 输出层（语言模型头）

    def forward(self, x, mask=None):
        # Step 1: 输入嵌入
        x = self.embedding(x)

        # Step 2: 依次通过多个 Transformer Block
        for block in self.blocks:
            x = block(x, mask)

        # Step 3: 语言模型头
        logits = self.lm_head(x)

        return logits

    def predict(self, input_seq, max_length=20, temperature=1.0, top_k=None):
        """
        用于生成序列的预测函数
        :param input_seq: 初始输入序列 (batch_size, seq_len)
        :param max_length: 生成的最大序列长度
        :param temperature: 控制采样分布的温度参数
        :param top_k: 限制生成候选的 top-k 采样
        :return: 生成的完整序列
        """
        self.eval()  # 设置模型为推理模式
        with torch.no_grad():
            for _ in range(max_length):
                seq_len = input_seq.size(1)

                # 因果掩码
                mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(input_seq.device)

                # 前向传播获取 logits
                logits = self.forward(input_seq, mask)

                # 调整 logits 分布的温度
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)

                # Top-k 采样（如果指定）
                if top_k is not None:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)  # 取 top-k
                    probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)  # 将非 top-k 的概率置 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)  # 重新归一化

                # 从分布中采样下一个 token
                next_token = torch.multinomial(probs, num_samples=1)

                # 将新 token 拼接到输入序列
                input_seq = torch.cat([input_seq, next_token], dim=-1)

        return input_seq

