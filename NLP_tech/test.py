import torch
from GPT2 import GPT2Embedding,MultiHeadAttention,AddNorm,FeedForwardNetwork,GPT2
import torch.nn as nn

def test_GPT2Embedding():
    # 定义 GPT2 嵌入层
    vocab_size = 10
    embed_dim = 4
    max_seq_len = 10
    gpt2_embedding = GPT2Embedding(vocab_size, embed_dim, max_seq_len)

    # 输入张量
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 形状 (batch_size=2, seq_len=3)

    # 获取最终嵌入
    final_embeddings = gpt2_embedding(x)
    print('Test GPT2Embedding')
    print("输入张量:\n", x)
    print("最终嵌入:\n", final_embeddings)
    print("嵌入形状:", final_embeddings.shape)  # 应为 (batch_size, seq_len, embed_dim)
    print('\n')

def test_MultiHeadAttention():
    # 模拟输入数据
    batch_size = 2
    seq_len = 4
    embed_dim = 8
    num_heads = 2

    # 输入张量
    Q = torch.rand(batch_size, seq_len, embed_dim)
    K = torch.rand(batch_size, seq_len, embed_dim)
    V = torch.rand(batch_size, seq_len, embed_dim)
    mask = torch.ones(batch_size, seq_len, seq_len)  # 全掩码，无屏蔽

    # 定义 Multi-Head Attention 模块
    multihead_attn = MultiHeadAttention(embed_dim, num_heads)

    # 前向传播
    output, attention_weights = multihead_attn(Q, K, V, mask)

    print('Test MultiHeadAttention')
    print("输出形状:", output.shape)  # (batch_size, seq_len, embed_dim)
    print("注意力权重形状:", attention_weights.shape)  # (batch_size, num_heads, seq_len, seq_len)
    print('\n')

def test_AddNorm():
    print('Test AddNorm')
    # 初始化 AddNorm
    embed_dim = 8  # 嵌入维度大小
    add_norm = AddNorm(embed_dim, eps=1e-6)

    # 模拟输入张量和子层输出
    batch_size, seq_len = 2, 4
    x = torch.rand(batch_size, seq_len, embed_dim)  # 原始输入张量
    sublayer_output = torch.rand(batch_size, seq_len, embed_dim)  # 子层输出张量

    print("\n原始输入张量 x:\n", x)
    print("子层输出张量 sublayer_output:\n", sublayer_output)

    # 前向传播
    output = add_norm(x, sublayer_output)
    print("\nAddNorm 输出张量:\n", output)

    # 测试形状一致性
    assert output.shape == x.shape, "输出形状与输入形状不一致！"

    # 测试残差连接（检查输出是否包含 x 和 sublayer_output 的信息）
    residual = x + sublayer_output  # 残差连接
    mean_residual = residual.mean(dim=-1, keepdim=True)  # 每个样本的均值
    std_residual = residual.std(dim=-1, keepdim=True)  # 每个样本的标准差
    normalized_residual = (residual - mean_residual) / (std_residual + 1e-6)  # 手动归一化
    normalized_output = (output - output.mean(dim=-1, keepdim=True)) / output.std(dim=-1, keepdim=True)

    # 检查手动归一化结果与 AddNorm 输出是否一致
    assert torch.allclose(normalized_residual, normalized_output, atol=1e-5), "归一化结果不一致！"

    print("\n测试通过：AddNorm 实现正确！")
    print('\n')

def test_feed_forward_network():
    print("=== 测试 FeedForwardNetwork ===")

    # 定义输入参数
    embed_dim = 8  # 输入嵌入维度
    hidden_dim = 32  # 隐藏层维度
    batch_size = 2
    seq_len = 4

    # 测试不同激活函数
    for activation in [nn.ReLU, nn.GELU]:
        print(f"\n测试激活函数: {activation.__name__}")

        # 初始化 FeedForwardNetwork
        ffn = FeedForwardNetwork(embed_dim, hidden_dim, activation=activation)

        # 模拟输入张量
        x = torch.rand(batch_size, seq_len, embed_dim)
        print("输入张量形状:", x.shape)

        # 前向传播
        output = ffn(x)
        print("输出张量形状:", output.shape)

        # 验证输入和输出形状一致
        assert output.shape == x.shape, "输出形状与输入形状不一致！"

        # 验证隐藏层是否有效
        hidden_output = ffn.fc1(x)
        assert hidden_output.shape == (batch_size, seq_len, hidden_dim), "隐藏层输出形状不正确！"

        # 打印测试通过信息
        print(f"激活函数 {activation.__name__} 测试通过！")

def generate_sequence(model, input_seq, max_length=50):
    """
    使用 GPT2 生成单词序列
    :param model: GPT2 模型
    :param input_seq: 输入序列 (batch_size, seq_len)
    :param max_length: 最大生成长度
    :return: 生成的序列
    """
    model.eval()
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_seq)  # 模型预测
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # 获取最后一个位置的预测
            input_seq = torch.cat([input_seq, next_token], dim=-1)  # 拼接预测的 token
    return input_seq


test_GPT2Embedding()
test_MultiHeadAttention()
test_AddNorm()
test_feed_forward_network()

# 模型参数
vocab_size = 50257
embed_dim = 768
num_heads = 12
hidden_dim = 3072
num_layers = 12
max_seq_len = 1024

# 初始化 GPT2 模型
model = GPT2(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len)

# 输入
batch_size = 2
seq_len = 20
x = torch.randint(0, vocab_size, (batch_size, seq_len))  # 随机生成 token 序列

# 因果掩码
batch_size, seq_len = x.size(0), x.size(1)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)  # 增加 batch 和 heads 维度
mask = mask.repeat(batch_size, num_heads, 1, 1)  # 扩展为 (batch_size, num_heads, seq_len, seq_len)


logits = model(x, mask)
print("输入形状:", x.shape)       # torch.Size([2, 20])
print("输出形状:", logits.shape)  # torch.Size([2, 20, 50257])

input_seq = torch.randint(0, vocab_size, (1, 5))  # 初始化输入序列
output_seq = generate_sequence(model, input_seq, max_length=10)
print("生成序列:", output_seq)