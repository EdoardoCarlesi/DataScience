from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

model = LinearRegression(normalize=True)
#print(model)

(X, y) = np.arange(10).reshape(5, 2), range(5)
print(X)
print(list(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_train, X_test)

print(y_train, y_test)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Supervised estimator: model.predict_proba() // model.score() 

# Unsupervised estimators: model.predict() // model.transform()

'''
 Algorithms:

    - Classification
    - Clusering
    - Regression
    - Dimensionality reduction

'''
