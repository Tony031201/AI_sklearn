import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# 1. load the data required
data = pd.read_csv('wdbc.data',header=None)

columns = ['ID','Diagnosis'] + [f'Fature {i}' for i in range(1,31)]

data.columns = columns

print('data head: ')
print(data.head())
print('data describe: ')
print(data.describe())

print(data['Diagnosis'].value_counts())

# data preprocessing
X = data.iloc[:,2:].values
Y = data['Diagnosis'].apply(lambda x: 1 if x =='M' else 0).values

# normalize
scaler = StandardScaler()
x_scale = scaler.fit_transform(X)

# divide into test set and train set
X_train,X_test,y_train,y_test = train_test_split(x_scale,Y,test_size=0.2,random_state=42)

# train GNB model
model = GaussianNB()
model.fit(X_train, y_train)

# predicate
y_pred = model.predict(X_test)

# info
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
