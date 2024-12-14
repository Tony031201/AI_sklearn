from datasets import load_dataset
from transformers import GPT2Tokenizer
import sys
from torch.utils.data import Dataset, DataLoader
import torch
from GPT2 import GPT2
from transformers import GPT2LMHeadModel
import torch.nn as nn
from transformers import AdamW

dataset = load_dataset("squad")
# print(dataset)
print("样本结构:", dataset['train'][0])

# preprocessing
# GPT2分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def preprocessing(example):
    # print("Now print")
    # print(example)
    try:
        context = example['context']

        question = example['question']
        answer = example["answers"]['text'] if example["answers"]['text'] else '无答案'

        # 拼接输入和目标
        input_text = f"上下文：{context} 问题：{question}"
        target_text = f"答案：{answer}"

        # 编码为 Token ID
        input_ids = tokenizer.encode(input_text, truncation=True, max_length=1024)
        target_ids = tokenizer.encode(target_text, truncation=True, max_length=1024)

        return {"input_ids": input_ids, "target_ids": target_ids}

    except Exception as e:
        print('Error occurs')
        print('question:',example['question'])
        print('answer1101:',example['answers'])
        sys.exit(1)

# 清洗数据
train_dataset = dataset["train"].map(preprocessing, batched=False)
validation_dataset = dataset["validation"].map(preprocessing, batched=False)

# 将Hugging Face 的 Dataset 转换为 PyTorch 的 Dataset 和 DataLoader 以便于训练
class GPT2Dataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids = self.dataset[idx]["input_ids"]
        target_ids = self.dataset[idx]["target_ids"]
        return torch.tensor(input_ids), torch.tensor(target_ids)

# 转换为 PyTorch Dataset
train_data = GPT2Dataset(train_dataset)
val_data = GPT2Dataset(validation_dataset)

# 定义 DataLoader
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4)

# 实现GPT2 模型
model = GPT2(
    vocab_size=50257,
    embed_dim=768,
    num_heads=12,
    hidden_dim=3072,
    num_layers=12,
    max_seq_len=1024
).to("cuda")

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 模型训练
# 训练循环
epochs = 3
model.train()

for epoch in range(epochs):
    total_loss = 0

    for input_ids, target_ids in train_loader:
        input_ids, target_ids = input_ids.to("cuda"), target_ids.to("cuda")

        # 前向传播
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            input_ids, target_ids = input_ids.to("cuda"), target_ids.to("cuda")

            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(val_loader)

# 调用验证函数
val_loss = evaluate(model, val_loader)
print(f"Validation Loss: {val_loss}")

def generate_answer(model, tokenizer, context, question, max_length=50):
    model.eval()

    input_text = f"上下文：{context} 问题：{question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")

    # 使用模型生成答案
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, temperature=0.7, top_k=5)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 测试生成
context = "The University of Notre Dame is a private Catholic university."
question = "What type of university is Notre Dame?"

answer = generate_answer(model, tokenizer, context, question)
print("生成的答案:", answer)
