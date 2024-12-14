import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim

# data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 MNIST 数据集
# torchvision.datasets.MNIST(
#     root='./data',        # 数据存储路径
#     train=True,           # 是否加载训练集（True）或测试集（False）
#     transform=transform,  # 数据预处理方法
#     download=True         # 如果本地没有数据，是否从网上下载
# )
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

# DataLoader
# 将数据集分成批次 每次加载64个样本，shuffle是打乱次序
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(train_dataset)

# 可视化数据
data_iter = iter(train_loader)
images, labels = next(data_iter)

# fig, axes = plt.subplots(1, 6, figsize=(12, 4))
# for i in range(6):
#     axes[i].imshow(images[i].squeeze(), cmap='gray')
#     axes[i].set_title(f"Label: {labels[i].item()}")
#     axes[i].axis('off')
# plt.show()

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 输入通道 1，输出通道 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 输入通道 32，输出通道 64
        # 最大池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 展平后的输入大小为 64 * 5 * 5
        self.fc2 = nn.Linear(128, 10)  # 输出层

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 第一卷积层
        x = self.pool(x)  # 第一池化层
        x = torch.relu(self.conv2(x))  # 第二卷积层
        x = self.pool(x)  # 第二池化层
        x = x.view(-1, 64 * 5 * 5)  # 展平
        x = torch.relu(self.fc1(x))  # 全连接层
        x = self.fc2(x)  # 输出层
        return x

    # 实例化模型


model = CNNModel()
# print(cnn_model)

## 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 4. 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新权重

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# 5. 测试模型
model.eval()  # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")