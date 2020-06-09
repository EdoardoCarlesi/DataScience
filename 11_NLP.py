import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import string

# Required to download extra files & datasets
#nltk.download_shell()

def text_process(mess):
    # Remove punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # Remove frequent/useless words
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return clean


data = 'dati/smsspamcollection/SMSSpamCollection'
#messages = [line.rstrip() for line in open(data)]

messages = pd.read_csv(data, sep='\t', names=['label','message'])
#print(messages.groupby('label').describe())

messages['length'] = messages['message']
#print(messages['message'].head(10).apply(text_process))

print('Vectorizing messages... ')
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print('Done. Total number of words: ', len(bow_transformer.vocabulary_))

mess = messages['message'][109]
#print(mess)

# Transform all the messages - Bag-of-Words - into the sparse matrix
messages_bow = bow_transformer.transform(messages['message'])
bow = bow_transformer.transform([mess])
#print(bow)

# TF: Term frequency, in the document (how often the given word appears)
# IDF: Inverse domain frequency - how many times in the "test" body of documents a given term is found (properly normalized)
print('TF-IDF Transformation...')
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
tfidf = tfidf_transformer.transform(bow)

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
print('Done. Resuts for a single random message:')
print('-> predicted:', spam_detect_model.predict(tfidf)[0])
print('-> expected:', messages.label[3])

all_predictions = spam_detect_model.predict(messages_tfidf)
print (classification_report(messages['label'], all_predictions))

plt.show()
