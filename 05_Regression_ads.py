from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

data_train = 'dati/advertising.csv'

train = pd.read_csv(data_train)
sns.set_style('whitegrid')

#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#sns.distplot(train['Age'], kde= False)
#sns.distplot(train['Daily Time Spent on Site'], kde= False)
#sns.jointplot( x = 'Age', y = 'Area Income', data = train)
#sns.jointplot( x = 'Age', y = 'Daily Time Spent on Site', data = train, kind = 'kde')
#sns.pairplot(data = train, hue = 'Clicked on Ad')

print(train.head(0))

train.drop(['Country', 'City', 'Ad Topic Line', 'Timestamp'], axis = 1, inplace=True)
X = train.drop('Clicked on Ad', axis = 1)
y = train['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

logmod = LogisticRegression()
logmod.fit(X_train, y_train)

predic = logmod.predict(X_test)

print(confusion_matrix(y_test, predic))
print(classification_report(y_test, predic))

'''
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
train.drop('Cabin', axis=1, inplace= True)
train.dropna(inplace = True)
#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#print(train.head())
#print(pd.get_dummies(train['Sex'], drop_first = True))
sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)
#print(embark)
train = pd.concat([train, sex, embark], axis = 1)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
#print(train.head())
X = train.drop('Survived', axis = 1)
y = train['Survived']
logmod = LogisticRegression()
logmod.fit(X_train, y_train)
print(X_test)
predic = logmod.predict(X_test)
print(classification_report(y_test, predic))
print(confusion_matrix(y_test, predic))
'''
plt.show()

