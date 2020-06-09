from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style('whitegrid')

#data_train = 'dati/Classified Data'
data_train = 'dati/KNN_Project_Data'

train = pd.read_csv(data_train, index_col = 0)

print(train.info())
print(train.head())

#sns.pairplot(train, hue='TARGET CLASS')

scaler = StandardScaler()
logmod = LogisticRegression()

knn = KNeighborsClassifier(n_neighbors = 37)

scaler.fit(train.drop('TARGET CLASS', axis = 1))
scaled_feat = scaler.transform(train.drop('TARGET CLASS', axis = 1))
data_scaled = pd.DataFrame(scaled_feat, columns = train.columns[:-1])

print(scaled_feat)

print(data_scaled.head())

X = data_scaled
y = train['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

knn.fit(X_train, y_train)
logmod.fit(X_train, y_train)

pred = knn.predict(X_test)
pred2 = logmod.predict(X_test)
print(classification_report(y_test, pred))
print(classification_report(y_test, pred2))

error_rate = []

for k in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    #print(k, '--------- \n', classification_report(y_test, pred))

    error_rate.append(np.mean(pred != y_test))

#print(error_rate)

plt.figure(figsize=(10,6))

plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed')
'''
'''

plt.show()





