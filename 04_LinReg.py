from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from sklearn import metrics

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

lm = LinearRegression()

df = pd.read_csv('dati/USA_Housing.csv')

'''
print(df.head())
print(df.describe())
print(df.info())
#sns.distplot(df['Price'])
#sns.heatmap(df.corr(), annot=True)
#print(df.columns)
#X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
'''

X = df[df.columns[:5]]
y = df['Price']


X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 101)
lm.fit(X_train, y_train)

#print(lm.intercept_)
#cdf = pd.DataFrame(lm.coef_, X.columns, columns = ['Coeff'])
#print(cdf)
#boston = load_boston()
#print(boston.keys())
#print(boston)

predictions = lm.predict(X_test)

#plt.scatter(y_test, predictions)
#sns.distplot(y_test-predictions)


mae = metrics.mean_absolute_error(y_test, predictions)
msq = metrics.mean_squared_error(y_test, predictions)
rme = np.sqrt(msq)

print(mae, msq, rme)


plt.show()
