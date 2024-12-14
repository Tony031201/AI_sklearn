from datasets import load_dataset
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
def clean(data):
    # clean the text
    cleaned_text  = re.sub(r"<.*?>", '', data['text'])
    cleaned_text = re.sub(r'[^a-zA-Z]',' ',cleaned_text).lower().strip()
    return {'text': cleaned_text}

dataset = load_dataset("tweet_eval", "sentiment")

train_data = dataset['train']
test_data = dataset['test']

# print(train_data[:1])
train_data = train_data.map(clean)
test_data = test_data.map(clean)

# print(train_data[:1])
train_texts = [example['text'] for example in train_data]
test_texts = [example['text'] for example in test_data]
train_labels = [example['label'] for example in train_data]
test_labels = [example['label'] for example in test_data]

tfidfVectorizer = TfidfVectorizer(max_features=10000)
X_train = tfidfVectorizer.fit_transform(train_texts)
X_test = tfidfVectorizer.transform(test_texts)
#
# # print(X_train[:1])
model = MultinomialNB()
model.fit(X_train,train_labels)
y_pred = model.predict(X_test)

# estimate the model
print("Accuracy:", accuracy_score(test_labels, y_pred))
print("Classification Report:\n", classification_report(test_labels, y_pred))