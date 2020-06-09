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

def convert(string):
    if string == 'Yes':
        return 1
    else:
        return 0

data = pd.read_csv('dati/College_Data')

print(data.info())
print(data.head())

#sns.scatterplot(x='Grad.Rate', y='Room.Board', data = data, hue = 'Private')
#sns.scatterplot(x='F.Undergrad', y='Outstate', data = data, hue = 'Private')


#data['Grad.Rate']['Cazenovia College'] = 100

#df['Event'].mask(df['Event'] == 'Hip-Hop', 'Jazz', inplace=True)

data['Grad.Rate'].mask(data['Grad.Rate'] > 100, 100, inplace=True)

#data[data['Private'] == 'No']['Grad.Rate'].hist(color='blue', bins=30)
#data[data['Private'] == 'Yes']['Grad.Rate'].hist(color='red', bins=30, alpha=0.5)

kmeans = KMeans(n_clusters = 2)
data_new = data.drop(['Unnamed: 0', 'Private'], axis=1)

kmeans.fit(data_new) 
#print(kmeans.cluster_centers_)
#print(kmeans.labels_)
data['Cluster'] = data['Private'].apply(convert)

print(classification_report(data['Cluster'], kmeans.labels_))
print(confusion_matrix(data['Cluster'], kmeans.labels_))




#print(data['Cluster'])
#print(data.head())

'''
# KMeans is unsupervised learning


fig, (ax1, ax2) = plt.subplots(1,  2, sharey = True, figsize = (10, 6))

ax1.set_title('K Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c = kmeans.labels_)

ax2.set_title('Original')
ax2.scatter(data[0][:, 0], data[0][:, 1], c = data[1], cmap = 'rainbow')
'''

plt.show()
