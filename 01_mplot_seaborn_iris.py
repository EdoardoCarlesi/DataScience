import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

'''
iris = sns.load_dataset('iris')
print(iris.head())

print(iris['species'].unique())

g = sns.PairGrid(iris)
#g.map(plt.scatter)

g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
'''

tips = sns.load_dataset('tips')

g = sns.FacetGrid(data=tips, col='time', row='smoker')

#g.map(sns.distplot, 'total_bill')
g.map(plt.scatter, 'total_bill', 'tip')

plt.show()
