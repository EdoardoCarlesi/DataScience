from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string
import pickle


data = 'dati/DATA/fake_reg.csv'

df = pd.read_csv(data)

print(df.head())

#sns.pairplot(df)
#plt.show()

X = df[['feature1','feature2']].values
y = df[['price']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

#print(X_train.shape)

# Normalize / scale data
scaler = MinMaxScaler()

# This only optimizes the parameters to perform the scaling later on
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create the neural network; units = number of neurons, activation = activation function
#model = Sequential([Dense(4, activation = 'relu'), Dense(2, activation = 'relu'), Dense(1)]) # ReLu = Rectified linear unit

# Alternative mode to create a NNW
model = Sequential()

# Dense() =  densely connected = normal feed forward neural network, where each network is connected to all the networks in the following layer
model.add(Dense(4, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse')

# Epochs - how many times you should go through the dataset
model.fit(x = X_train, y = y_train, epochs = 250, verbose = 0)

# Loss function evaluation as a function of the step
loss_df = pd.DataFrame(model.history.history)

# Metric loss of the model
loss = model.evaluate(X_test, y_test)

print('MSE : ', loss)

# Predict the new prices
pred = model.predict(X_test)

test_predictions = pd.Series(pred.reshape(300,))

pred_df = pd.DataFrame(y_test, columns= ['True Y'])
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['TrueY', 'Prediction']

sns.scatterplot(x=pred_df['TrueY'], y=pred_df['Prediction'])

# model.save('my_model.h5')
# later_model = load('my_model.h5')

#print(pred_df)

plt.show()
#exit()




