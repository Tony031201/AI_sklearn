import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# load data set
data = pd.read_csv('winequality-white.csv',sep=';',header=0)
data_red = pd.read_csv('winequality-red.csv',sep=';',header=0)

# DATA preprocessing
# print('data head: ')
# print(data.head())
# print('data describe: ')
# print(data.describe())
X_white = data.iloc[:,:11].values
X_red = data_red.iloc[:,:11].values
X_combined = np.vstack((X_white, X_red))

Y = []
for _ in range(len(X_white)):
    Y.append(1)

for _ in range(len(X_red)):
    Y.append(2)

print('X_combined:')
print(X_combined)
print('Y:')
print(Y)

# normalize
scaler = StandardScaler()
x_scale = scaler.fit_transform(X_combined)

# divide into test set and train set
X_train,X_test,y_train,y_test = train_test_split(x_scale,Y,test_size=0.2,random_state=42)



# load the GMM model
# plan to divide into several clusters
gmm = GaussianMixture(n_components=2,covariance_type='full',random_state=42)

# train
gmm.fit(X_train)

cluster_labels = gmm.predict(X_test)
print("Cluster labels for the first 10 samples:")
print(cluster_labels[:10])

sil_score = silhouette_score(X_test, cluster_labels)
print(f"Silhouette Score: {sil_score}")
#
ari_score = adjusted_rand_score(y_test, cluster_labels)
print(f"Adjusted Rand Index (ARI): {ari_score}")

print('After apply LDA')
# if apply LDA
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# train
gmm.fit(X_train_lda)

cluster_labels = gmm.predict(X_test_lda)

sil_score = silhouette_score(X_test_lda, cluster_labels)
print(f"Silhouette Score: {sil_score}")
#
ari_score = adjusted_rand_score(y_test, cluster_labels)
print(f"Adjusted Rand Index (ARI): {ari_score}")



