from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from sklearn import metrics

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

lm = LinearRegression()

df = pd.read_csv('dati/ecommerce_customers')

print(df.info())
print(df.describe())
print(df.head(5))

#sns.jointplot(x = df['Time on App'], y = df['Yearly Amount Spent'], kind = 'hex')
#sns.lmplot(x = 'Time on Website', y = 'Yearly Amount Spent', data = df)
sns.lmplot(x = 'Time on App', y = 'Yearly Amount Spent', data = df)
sns.pairplot(df)

X = df[df.columns[3:7]]
y = df['Yearly Amount Spent']

X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
lm.fit(X_train, y_train)

print(lm.coef_)

pred = lm.predict(X_test)

print(pred)

#plt.scatter(y_test, pred)
sns.distplot(y_test -pred)

plt.show()
