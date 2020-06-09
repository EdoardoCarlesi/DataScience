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

sns.set_style('white')

titles = 'dati/Movie_Id_Titles'
data = 'dati/u.data'
item = 'dati/u.item'

columns_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv(data, sep='\t', names = columns_names)
#print(df.head())

movie_titles = pd.read_csv(titles)
df = pd.merge(df, movie_titles, on = 'item_id')
#print(df.head())

mean_rating = df.groupby('title')['rating'].mean()
count_rating = df.groupby('title')['rating'].count()

ratings = pd.DataFrame(mean_rating)
ratings['n_ratings'] = count_rating

#ratings['n_ratings'].hist(bins=50) 
#ratings['rating'].hist(bins=50) 
#sns.jointplot(x='rating', y='n_ratings', data = ratings)

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

ratings.sort_values('n_ratings', ascending=False).head(10)

starwars_ratings = moviemat['Star Wars (1977)']
liarliar_ratings = moviemat['Liar Liar (1997)']

#print(starwars_ratings.head())
#print(moviemat.corrwith(starwars_ratings))

starwars_related = moviemat.corrwith(starwars_ratings)
liarliar_related = moviemat.corrwith(liarliar_ratings)

corr_sw = pd.DataFrame(starwars_related, columns=['Correlation'])
corr_sw.dropna(inplace=True)

corr_sw = corr_sw.join(ratings['n_ratings'])

print(corr_sw[corr_sw['n_ratings'] > 15].sort_values('Correlation', ascending=False).head(20))

#print(moviemat)
#print(ratings)
plt.show()

