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

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    avg = [37, 29, 24]

    if pd.isnull(Age):

        if Pclass == 1:
            return avg[0]
        if Pclass == 2:
            return avg[1]
        if Pclass == 3:
            return avg[2]
    else:
        return Age




data_train = 'dati/titanic_train.csv'
data_test = 'dati/titanic_test.csv'

train = pd.read_csv(data_train)

#print(train.info())
#print(train.isnull())

#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.set_style('whitegrid')
#sns.countplot(x='Survived', data=train, hue='Pclass')
#sns.distplot(train['Age'], kde=False, bins=25)
#train['Age'].plot.hist(bins=20)
#sns.countplot(x='SibSp', data=train)
#train['Fare'].plot.hist(bins=40, figsize=(10,4))

#sns.boxplot(x='Pclass', y='Age', data = train)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

logmod = LogisticRegression()
logmod.fit(X_train, y_train)

print(X_test)

predic = logmod.predict(X_test)

print(classification_report(y_test, predic))
print(confusion_matrix(y_test, predic))

plt.show()

