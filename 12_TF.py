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
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string
import numpy as np

def convert_status(string):
    if string == 'Fully Paid':
        return 1
    else:
        return 0


data_info = 'dati/DATA/lending_club_info.csv'
data_loan = 'dati/DATA/lending_club_loan_two.csv'

df_info = pd.read_csv(data_info)
df_loan_orig = pd.read_csv(data_loan)

df_loan = df_loan_orig.sample(frac = 0.2, random_state = 69)

print(df_loan.head())
print(df_loan.info())
#print(df_loan.describe().transpose())
#print(df_loan['loan_status'])
#print(df_loan.corr())
#print(df_loan['loan_status'])
#print(df_loan['mort_acc'])
#print(df_loan['pub_rec_bankruptcies'])
#print(df_loan['earliest_cr_line'])
#print(len(df_loan['title'].value_counts() > 10))

df_loan['earliest_cr_line'] = pd.to_datetime(df_loan['earliest_cr_line'])
df_loan['year_cr_line'] = df_loan['earliest_cr_line'].apply(lambda date : date.year)
df_loan['loan_status'] = df_loan['loan_status'].apply(convert_status)


'''
DATA VISUALIZATION
'''

#print(df_loan['year_cr_line'])
#sns.heatmap(df_loan.corr())
#subgrade_ord = sorted(df_loan['sub_grade'].unique())
#print(subgrade_ord)
#print(df_loan.groupby('loan_status')['loan_amnt'].mean())
#plt.figure(figsize=(12,8))
#print(df_loan['sub_grade'])
#sns.countplot(x='sub_grade', data= df_loan, order = subgrade_ord, hue='loan_status')
#df_fg = df_loan[(df_loan['grade'] == 'F') | (df_loan['grade'] == 'G')]
#subgrade_ord = sorted(df_fg['sub_grade'].unique())
#plt.figure(figsize=(12,4))
#sns.countplot(x='sub_grade', data= df_fg, order = subgrade_ord, hue='loan_status')
#df_loan.corr()['loan_status'].sort_values().drop('loan_status').plot(kind='bar')

'''
DATA CLEANING
'''

n_elem = len(df_loan['loan_status'])
#print('Total elements: ', n_elem)

df_loan.drop('emp_title', axis = 1, inplace = True)
#print(df_loan.isnull().sum() / n_elem)

emp_length = df_loan['emp_length'].unique()
#print(emp_length)

ord_emp_len = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
#sns.countplot(x='emp_length', order=ord_emp_len, data=df_loan, hue='loan_status')

rate_emp_len = []

for emp_len in ord_emp_len:
    unpaid = df_loan[(df_loan['emp_length'] == emp_len) & (df_loan['loan_status'] == 0)]
    all_loans = df_loan[(df_loan['emp_length'] == emp_len)]
    rate_emp_len.append(len(unpaid)/len(all_loans))
    #    print(emp_len, ' : ', len(all_loans), ' unpaid: ', len(unpaid))

#print(df_loan.corr()['total_acc'].sort_values())
#print(df_loan.corr()['mort_acc'].sort_values())

df_loan.drop('title', axis = 1, inplace = True)
df_loan.drop('grade', axis = 1, inplace = True)
df_loan.drop('emp_length', axis = 1, inplace = True)

tot_acc_avg = df_loan.groupby('total_acc').mean()['mort_acc']

def fix_mort(mort, tot):
    if np.isnan(mort):
        return tot_acc_avg[tot]
    else:
        return mort

df_loan['mort_acc'] = df_loan.apply(lambda x: fix_mort(x['mort_acc'], x['total_acc']), axis = 1)

df_loan.dropna(inplace = True)
#print(df_loan.isnull().sum() / n_elem)

df_loan['term'] = df_loan['term'].apply(lambda term: int(term[:3]))

print(df_loan['term'])

subgrade_dummies = pd.get_dummies(df_loan['sub_grade'], drop_first=True)
df_loan = pd.concat([df_loan.drop('sub_grade',axis=1),subgrade_dummies],axis=1)

verification_dummies = pd.get_dummies(df_loan['verification_status'], drop_first=True)
df_loan = pd.concat([df_loan.drop('verification_status',axis=1), verification_dummies],axis=1)

application_dummies = pd.get_dummies(df_loan['application_type'], drop_first=True)
df_loan = pd.concat([df_loan.drop('application_type',axis=1), application_dummies],axis=1)

#initial_dummies = pd.get_dummies(df_loan['initial_list_status'], drop_first=True)
#df_loan = pd.concat([df_loan.drop('initial_list_status',axis=1), initial_dummies],axis=1)
#print(initial_dummies)

purpose_dummies = pd.get_dummies(df_loan['purpose'], drop_first=True)
df_loan = pd.concat([df_loan.drop('purpose',axis=1), purpose_dummies],axis=1)

df_loan['home_ownership'] = df_loan['home_ownership'].replace(['ANY', 'NONE'], 'OTHER')
ownership_dummies = pd.get_dummies(df_loan['home_ownership'], drop_first=True)
df_loan = pd.concat([df_loan.drop('home_ownership',axis=1), ownership_dummies],axis=1)

df_loan['zip_code'] = df_loan['address'].apply(lambda address:address[-5:])
zip_code_dummies = pd.get_dummies(df_loan['zip_code'], drop_first=True)
df_loan = pd.concat([df_loan.drop('zip_code',axis=1), zip_code_dummies],axis=1)

# Drop the last lines
df_loan.drop('initial_list_status', axis=1, inplace=True)
df_loan.drop('address', axis=1, inplace=True)
df_loan.drop('issue_d', axis=1, inplace=True)
df_loan.drop('earliest_cr_line', axis = 1, inplace = True)

print(df_loan.columns)
n_feat = len(df_loan.columns)
print('Total of ', len(df_loan.columns), ' features.')

X = df_loan.drop('loan_status', axis= 1).values
y = df_loan['loan_status'].values

index = 3243
X_rand = X[index]
new_cust = X_rand.reshape(1, n_feat-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Normalize / scale data
scaler = MinMaxScaler()

# This only optimizes the parameters to perform the scaling later on
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

n_layer2 = n_feat #39 #int(n_feat / 2)
n_layer3 = n_feat #19 #int(n_feat / 4)
n_epochs = 20
n_patience = 5

# Dense() =  densely connected = normal feed forward neural network, where each network is connected to all the networks in the following layer
model.add(Dense(n_feat, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(n_layer2, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(n_layer3, activation = 'relu'))
model.add(Dropout(0.7))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', mode="min", verbose = 1, patience = n_patience) 
# Epochs - how many times you should go through the dataset
model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test), epochs = n_epochs, callbacks=[early_stop])

# Loss function evaluation as a function of the step
losses = pd.DataFrame(model.history.history) 
losses.plot()

predictions = model.predict_classes(X_test)

print(classification_report(predictions, y_test))
print(confusion_matrix(predictions, y_test))

pred = model.predict_classes(new_cust) 
print(pred, df_loan.iloc[index]['loan_status'])

plt.show()



