from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns

import matplotlib.pyplot as plt
import cufflinks as cf
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#from plotly import __version__

#init_notebook_mode(connected = True)
cf.go_offline()


start = dt.datetime(2006, 1, 1)
end = dt.datetime(2016, 1, 1)

BAC = data.DataReader("BAC", 'yahoo', start, end)
# CitiGroup
C = data.DataReader("C", 'yahoo', start, end)
# Goldman Sachs
GS = data.DataReader("GS", 'yahoo', start, end)
# JPMorgan Chase
JPM = data.DataReader("JPM", 'yahoo', start, end)
# Morgan Stanley
MS = data.DataReader("MS", 'yahoo', start, end)
# Wells Fargo
WFC = data.DataReader("WFC", 'yahoo', start, end)

ticks = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

#print(BAC.info())

all_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=ticks)
all_stocks.columns.names = ['Ticks', 'Stocks']

#print(all_stocks.info())
#print(all_stocks.head())

returns = pd.DataFrame()

ast = all_stocks.xs(key='Close', axis=1, level='Stocks').max()
print(ast)

'''
#for tick in ticks:
#    returns[tick + ' return'] = all_stocks[tick]['Close'].pct_change()

print(returns.idxmax())
print(returns.std())
print(returns.info())
print(returns.head())

#sns.distplot(returns['2015-01-01':'2015-12-31']['MS return'], bins=50)
#print(returns.head())
#sns.pairplot(returns[1:])
'''

#plt.fig()
sns.set_style('whitegrid')

#for tick in ticks:
    #all_stocks[tick]['Close'].plot(figsize=(12,4), label=tick)
    #all_stocks[ticks]['Close'].plot(figsize=(12,4), label=tick)

plt.figure(figsize=(12,6))
BAC['Close']['2008-01-01':'2008-12-31'].rolling(window=10).mean().plot(label='10 Days Avg')
BAC['Close']['2008-01-01':'2008-12-31'].plot(label='Stock')
plt.legend()

plt.show()
