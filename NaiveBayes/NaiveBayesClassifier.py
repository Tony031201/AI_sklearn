import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = {}
        self.likelihood = {}
        self.margin = 0
        self.priors = {}
        self.features = {}

    def fit(self,X,y):
        self.features = X.columns
        self.classes = y.unique()
        self.priors = y.value_counts(normalize=True).to_dict()
        print('features:',self.features)
        print('classes:',self.classes)
        print('priors:',self.priors)

        # calculate likelihood P(A|C)
        self.likelihood = {} # clear the likelihood before
        for feature in self.features:
            self.likelihood[feature] = {}
            for c in self.classes:
                get_feature = X[y == c][feature]
                valueCount = get_feature.value_counts(normalize=True).to_dict()
                total_count = len(get_feature)
                unique_values = len(X[feature].unique())
                self.likelihood[feature][c] = {val: (valueCount.get(val, 0) + 1) / (total_count + unique_values)
                                               for val in X[feature].unique()}

    def predict(self,X):
        pred_list = []

        for _, row in X.iterrows():
            posteriors = {}
            for c in self.classes:
                posterior = self.priors[c]
                for feature in self.features:
                    likelihood = self.likelihood[feature][c].get(row[feature],0)
                    posterior *= likelihood
                posteriors[c] = posterior
            pred = max(posteriors,key=posteriors.get)
            pred_list.append(pred)

        return np.array(pred_list)
