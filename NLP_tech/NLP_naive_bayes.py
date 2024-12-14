import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def clean(text):
    # drop html label
    text['text'] = re.sub(r"<.*?>",'',text['text'])
    # drop non alphabet
    text['text'] = re.sub(r"[^a-zA-Z]",' ',text['text'])
    text['text'] = text['text'].lower().strip()
    return text

def main():
    # load data
    dataset = load_dataset("imdb")

    train_data = dataset['train']
    test_data = dataset['test']

    # preprocessing
    # print("TensorFlow Version:", tf.__version__)
    # print("Keras Version:", tf.keras.__version__)
    train_data = train_data.map(clean)
    test_data = test_data.map(clean)

    # print(train_data['text'][:2])
    # transfer the text to feature vector
    # initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_train = tfidf_vectorizer.fit_transform(train_data['text'])
    X_test = tfidf_vectorizer.transform(test_data['text'])
    # print(X_train[:1])

    model = MultinomialNB()
    # print(train_data['label'][:5])
    model.fit(X_train,train_data['label'])

    pred_y = model.predict(X_test)
    # estimate the model
    print("Accuracy:", accuracy_score(test_data['label'], pred_y))
    print("Classification Report:\n", classification_report(test_data['label'], pred_y))
if __name__ == '__main__':
    main()