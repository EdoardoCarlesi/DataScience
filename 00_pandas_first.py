import pandas as pd
import numpy as np
from numpy.random import randn


''' PT.1
#np.random.seed(101)
#df = pd.DataFrame(randn(3,3), ['Dio', 'Cane', 'Maiale'], ['A', 'B', 'C'])
#print(df)
#print(df[['A', 'B']])
#df['G'] = df['A'] + randn(1)
#df.drop('Dio', axis=0, inplace=True)
#print(df.loc['Dio'])
#print(df.iloc[1])
#print(df > 1)
#booldf = df > 1
#print(df[booldf])
#print(df['A'] > 0)
#print(df[df['A'] > 1])
#print(df[df['A'] > 0]['C'])
#df.reset_index(inplace=True)
#print(df)
'''

''' PT. 2
outside = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside, inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
#print(hier_index)
df = pd.DataFrame(randn(6,2), hier_index, ['A', 'B'])
df.index.names = ['Groups', 'Num']
print(df)
print(df.loc['G1'].loc[1])
#val = df.loc['G2'].loc[2]['B']
#val = df.xs('G1')
val = df.xs(1, level='Num')
print(val)
'''


''' PT. 3 

d = {'A':[1,2, np.nan], 'B':[5, np.nan, np.nan], 'C':[1, 2, 3]}

df = pd.DataFrame(d)

df.fillna(value=df['A'].mean(), inplace=True)

#print(df.dropna(thresh=2.1))
print(df)

'''


''' PT. 4

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])

'''

df = pd.DataFrame(randn(3,3), ['A', 'B', 'C'])
print(df)
#val = df[0].apply(lambda x: x * x)
val = df.loc['A'].apply(lambda x: np.sqrt(x * x)).sort_values()

print(val)

