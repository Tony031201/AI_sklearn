import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import movie_reviews
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# download the dateset from nltk
nltk.download('movie_reviews')

# load
reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

## data preprocessing
# transfer the data to DataFrame
data = pd.DataFrame(reviews,columns=['review','sentiment'])
# print(data.head())

# transfer the words to string
data['review'] = data['review'].apply(lambda x:''.join(x))

# divide the date into train set and test set
X = data['review']
Y = data['sentiment']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# apply CountVectorizer to transfer the string to the BoW
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# start to train the model
# first I wanna start with the naive_bayes
print('native-bayes model:')
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized,Y_train)

# predict
y_pred = nb_model.predict(X_test_vectorized)

# estimate the model
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))

# But what if I use LogisticRegression model?
print('LogisticRegression model:')
lr_model = LogisticRegression()
lr_model.fit(X_train_vectorized,Y_train)

# predict
y_pred = nb_model.predict(X_test_vectorized)

# estimate the model
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))

# extension
# try to replace BoW with TF-IDF
print('Replace Bow with TF-IDF')
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized,Y_train)

# predict
y_pred = nb_model.predict(X_test_vectorized)

# estimate the model
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))
