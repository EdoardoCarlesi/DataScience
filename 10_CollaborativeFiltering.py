from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style('white')

def predict(ratings, similarity, test_type='user'):
    
    if test_type == 'user':
        mean_rating = ratings.mean(axis = 1)

        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_rating[:, np.newaxis])
        pred = mean_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        return pred

    elif test_type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
        return pred


data = 'dati/u.data'
titles = 'dati/Movie_Id_Titles'
columns_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv(data, sep='\t', names = columns_names)

movie_titles = pd.read_csv(titles)
df = pd.merge(df, movie_titles, on = 'item_id')
#print(df.head())

n_users = df.user_id.nunique()
n_items = df.item_id.nunique()

print('Users: ', n_users, ' items: ', n_items)
train_data, test_data = train_test_split(df, test_size=0.25)

train_mat = np.zeros((n_users, n_items))
print(train_data.info())

print(train_data)

for line in train_data.itertuples():
#    print(line, line[1]-1, line[2]-1, line[3])
    train_mat[line[1]-1, line[2]-1] = line[3]

test_mat = np.zeros((n_users, n_items))
print(test_data.info())
for line in test_data.itertuples():
    test_mat[line[1]-1, line[2]-1] = line[3]

# User similarity computes the cosine of the angle of the vectors in the "ratings" space
user_similarity = pairwise_distances(train_mat, metric='cosine')
item_similarity = pairwise_distances(train_mat.T, metric='cosine')

#print(user_similarity)

item_pred = predict(train_mat, item_similarity, test_type='item')
user_pred = predict(train_mat, user_similarity, test_type='user')

#print('Item: ', item_pred)
#print('User: ', user_pred)



def rmse(prediction, ground_truth):
    print(ground_truth)
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print('Item: ', rmse(item_pred, test_mat)) #data))
print('User: ', rmse(user_pred, test_mat))

'''
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
'''
plt.show()

