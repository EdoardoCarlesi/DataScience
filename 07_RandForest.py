from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz

sns.set_style('whitegrid')
data_train = 'dati/kyphosis.csv'

train = pd.read_csv(data_train)

print(train.info())
print(train.head())

#sns.pairplot(train, hue = 'Kyphosis')

#scaler = StandardScaler()
#logmod = LogisticRegression()
#knn = KNeighborsClassifier(n_neighbors = 37)
dtree = DecisionTreeClassifier()

X = train.drop('Kyphosis', axis=1)
y = train['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 40)
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

'''
feat = list(train.columns[1:])

dot_data = StringIO()

export_graphviz(dtree, out_file = dot_data, feature_names = feat, rounded = True)
#print(dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
print(graph[0].create_ps())
Image(graph[0].create_png())

print(feat)

print('RANDOM FOREST')

rforest = RandomForestClassifier(n_estimators = 200)
rforest.fit(X_train, y_train)
pred2 = rforest.predict(X_test)

print(classification_report(y_test, pred2))
print(confusion_matrix(y_test, pred2))


scaler.fit(train.drop('TARGET CLASS', axis = 1))
scaled_feat = scaler.transform(train.drop('TARGET CLASS', axis = 1))
data_scaled = pd.DataFrame(scaled_feat, columns = train.columns[:-1])

print(scaled_feat)

print(data_scaled.head())

pred = knn.predict(X_test)
pred2 = logmod.predict(X_test)
print(classification_report(y_test, pred))

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

plt.show()





