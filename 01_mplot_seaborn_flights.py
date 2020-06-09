import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

flights = sns.load_dataset('flights')
print(flights.head())

tips = sns.load_dataset('tips')
#print(tips.head())


'''
    Heatmap
'''

#tc = tips.corr()
#print(tc) 
#sns.heatmap(tc, annot=True, cmap='coolwarm')

# Build a matrix out of a given dataframe
#fp = flights.pivot_table(index='month', columns='year', values='passengers')
#print(flights)
#print(fp)
#sns.heatmap(fp, linecolor='black', linewidths=1)
#sns.clustermap(fp)

plt.show()
