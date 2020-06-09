from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, explained_variance_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string
import pickle

def renew(year):
    if year > 1980:
        return 1
    else:
        return 0


data = 'dati/DATA/kc_house_data.csv'
df = pd.read_csv(data)

#print(df.head())
#print(df.isnull().sum())

print(df.info())
print(df.describe().transpose())

#plt.figure(figsize=(10, 6))
#sns.distplot(df['price'])
#sns.countplot(df['bedrooms'])

print(df.corr()['price'])

#sns.scatterplot(x='price', y='sqft_living', data=df)
#sns.scatterplot(x='long', y='lat', data=df, hue='price')
#sns.scatterplot(x='price', y='long', data=df)

#print(df.sort_values('price', ascending=False).head(210))

#sns.scatterplot(x='long', y='lat', data=bottom_99, hue='price', edgecolor=None, alpha=0.1)
#sns.boxplot(x='waterfront', y='price', data=bottom_99)

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

df.drop('id', axis=1, inplace=True)
df.drop('zipcode', inplace=True, axis=1)
df.drop('date', inplace=True, axis=1)

one_percent = int(len(df) * 0.01)
bottom_99 = df.sort_values('price', ascending=False).iloc[one_percent:]

#sns.boxplot(x='month', y='price', data=df)
#df.groupby('month').mean()['price'].plot()
#df.groupby('year').mean()['price'].plot()
#df.drop('date', axis=1, inplace=True)
#print(df.columns)

#df['renovated'] = df['yr_renovated'].apply(renew)
#df.drop('yr_renovated', inplace=True, axis=1)
#print(df['yr_renovated'].value_counts())

#X = df.drop('price', axis=1).values
#y = df['price'].values
X = bottom_99.drop('price', axis=1).values
y = bottom_99['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Normalize / scale data
scaler = MinMaxScaler()

# This only optimizes the parameters to perform the scaling later on
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

# 19 = number of layers = number of features
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

# Final layer - only outputs the predicted price
model.add(Dense(1)) 

model.compile(optimizer='adam', loss='mse')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# The validation data is used only to check while it is training but NOT to train the net or update weights and biases
model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test),
        batch_size = 128, epochs = 80)

losses = pd.DataFrame(model.history.history) #, columns = ['loss', 'val_loss'])
#losses.plot()

predictions = model.predict(X_test)

mse = mean_squared_error(predictions, y_test)
mae = mean_absolute_error(predictions, y_test)
evs = explained_variance_score(y_test, predictions)

print(mse, ' MSE: ', np.sqrt(mse), ' MAE: ', mae, ' EVS: ', evs)

plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')

plt.show()




