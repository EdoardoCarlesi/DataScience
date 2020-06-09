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

cancer = load_breast_cancer()

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df_target = pd.DataFrame(cancer['target'], columns=['Cancer'])

#cancer.keys()
#print(df.info())
#print(df.head())

X = df
y = df_target
X_df, X_test, y_df, y_test = train_test_split(X, np.ravel(y), test_size = 0.3, random_state = 40)

model = SVC()
model.fit(X_df, y_df)

pred = model.predict(X_test)

print(model)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

param_grid = {'C':[0.1, 1.0, 10, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=0)

grid.fit(X_df, y_df)

pps = grid.best_params_
print(pps)

pred = grid.predict(X_test)

print(grid)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))




'''
#sns.pairplot(df, hue = 'Kyphosis')
#scaler = StandardScaler()
#logmod = LogisticRegression()
#knn = KNeighborsClassifier(n_neighbors = 37)
feat = list(df.columns[1:])
dot_data = StringIO()
export_graphviz(dtree, out_file = dot_data, feature_names = feat, rounded = True)
#print(dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
print(graph[0].create_ps())
Image(graph[0].create_png())
print(feat)
print('RANDOM FOREST')
rforest = RandomForestClassifier(n_estimators = 200)
rforest.fit(X_df, y_df)
pred2 = rforest.predict(X_test)
print(classification_report(y_test, pred2))
print(confusion_matrix(y_test, pred2))
scaler.fit(df.drop('TARGET CLASS', axis = 1))
scaled_feat = scaler.transform(df.drop('TARGET CLASS', axis = 1))
data_scaled = pd.DataFrame(scaled_feat, columns = df.columns[:-1])
print(scaled_feat)
print(data_scaled.head())

pred = knn.predict(X_test)
pred2 = logmod.predict(X_test)
print(classification_report(y_test, pred))

error_rate = []

for k in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_df, y_df)
    pred = knn.predict(X_test)
    #print(k, '--------- \n', classification_report(y_test, pred))

    error_rate.append(np.mean(pred != y_test))

#print(error_rate)

plt.figure(figsize=(10,6))

plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed')
'''

plt.show()





