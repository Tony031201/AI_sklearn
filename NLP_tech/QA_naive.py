from datasets import load_dataset
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from clean import clean

# 加载 SQuAD 数据集
dataset = load_dataset("squad")

# 查看数据集结构
print(dataset)  # 显示数据集的基本信息
print(dataset['train'][2])  # 查看训练集中第一条样本