import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string
import pickle


def text_process(mess):
    # Remove punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # Remove frequent/useless words
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return clean


data = 'dati/yelp.csv'

reviews = pd.read_csv(data)
reviews['text length'] = reviews['text'].apply(len)

print(reviews.info())

#g = sns.FacetGrid(reviews, col='stars')
#g.map(plt.hist,'text length')

#sns.boxplot(x='stars', y='text length', data = reviews)
#reviews['stars'].hist()

#sns.countplot(x='stars', data=reviews)

stars = reviews.groupby('stars').mean()
#print(reviews.corr())
#sns.heatmap(stars.corr(), cmap='coolwarm', annot=True)

reviews_class = reviews[(reviews.stars==1) | (reviews.stars==5)]

X = reviews_class['text']
y = reviews_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

print('Pipeline ...')
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
#    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
#    ('classifier', RandomForestClassifier(n_estimators = 200)),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

'''
X = CountVectorizer().fit_transform(X)

#pickle.dump(x_transform, 'output/yelp_trans.pkl')
#print(y)


nb = MultinomialNB()
nb.fit(X_train, y_train)

pred = nb.predict(X_test)
'''

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


