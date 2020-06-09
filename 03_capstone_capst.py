import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def split_string(string):
    new_str = string.split(':', 1)
    #print(new_str[0])
    return new_str[0]


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

file911='/home/edoardo/Udemy/DataScience/Dispense/10-Data-Capstone-Projects/911.csv'

df = pd.read_csv(file911)

print(df.info())
print(df.head())
print(df.head(0))
#print(df['zip'].value_counts())
#print(df['twp'].value_counts())
#print(len(df['title'].apply(lambda x: split_string(x)))
df['Reason'] = df['title'].apply(lambda x: split_string(x))

#print(df['Reasons'].value_counts())
#sns.countplot(x = 'Reasons', data = df)

#print(type(df['timeStamp'].values[0]))
time = pd.to_datetime(df['timeStamp'].values)

df['Hour'] = time.hour
df['Month'] = time.month
#df['Day of the week'] = time.dayoftheweek

print((df['Hour']))
#print((df['Day of the week']))

sns.countplot(x='Month', hue = 'Reason', data = df)
#plt.legend()

plt.show()



