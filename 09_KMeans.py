from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_blobs

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


data = make_blobs(n_samples = 200, n_features = 2, centers = 4, cluster_std = 1.8, random_state = 101)
#plt.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap='rainbow')

# KMeans is unsupervised learning
kmeans = KMeans(n_clusters = 3)

kmeans.fit(data[0])

print(kmeans.cluster_centers_)
print(kmeans.labels_)

fig, (ax1, ax2) = plt.subplots(1,  2, sharey = True, figsize = (10, 6))

ax1.set_title('K Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c = kmeans.labels_)

ax2.set_title('Original')
ax2.scatter(data[0][:, 0], data[0][:, 1], c = data[1], cmap = 'rainbow')

plt.show()

