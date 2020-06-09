from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


sns.set_style('whitegrid')

iris = sns.load_dataset('iris')

print(iris.info())
print(iris.head())

#sns.pairplot(iris, hue='species')

iris['species'].unique()

cat_feats = ['species']
new_cat = pd.get_dummies(iris, columns=cat_feats, drop_first=True)

X = iris.drop('species', axis=1)
y = iris['species']

X_df, X_test, y_df, y_test = train_test_split(X, np.ravel(y), test_size = 0.3, random_state = 40)

model = SVC()
model.fit(X_df, y_df)
pred = model.predict(X_test)

#print(model)
#print(classification_report(y_test, pred))
#print(confusion_matrix(y_test, pred))

param_grid = {'C':[0.1, 1.0, 10, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=1)
grid.fit(X_df, y_df)
pps = grid.best_params_
#print(pps)
pred = grid.predict(X_test)

print(grid)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))





'''
#cancer.keys()
#print(df.info())
#print(df.head())
'''

plt.show()





