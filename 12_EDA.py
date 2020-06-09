from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, explained_variance_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string
import pickle

data = 'dati/DATA/cancer_classification.csv'
df = pd.read_csv(data)

print(df.info())
print(df.describe().transpose())

#sns.countplot(x='benign_0__mal_1', data=df)
#sns.heatmap(df.corr())

#df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')

X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Normalize / scale data
scaler = MinMaxScaler()

# This only optimizes the parameters to perform the scaling later on
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

# n_lay  = number of layers = number of features
n_lay = 30
model.add(Dense(n_lay, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_lay/2, activation='relu'))
model.add(Dropout(0.5))

# Final layer: Sigmoid for binary classification 0 or 1
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

print('Sanity check ')
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print('Done')

# If the metrics is accurracy then the mode is max (maximize) if the metrics is loss then we want to minimize (min)
early_stop = EarlyStopping(monitor='val_loss', mode="min", verbose = 1, patience = 25) 

# The validation data is used only to check while it is training but NOT to train the net or update weights and biases
model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test), epochs = 600, callbacks=[early_stop])

losses = pd.DataFrame(model.history.history) 
#losses.plot()

#predictions = np.rint(model.predict(X_test))
predictions = model.predict_classes(X_test)

print(classification_report(predictions, y_test))
print(confusion_matrix(predictions, y_test))

plt.show()

