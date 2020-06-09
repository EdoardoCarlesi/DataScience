import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

tips = sns.load_dataset('tips')

print(tips.head())

'''
    PLOT TYPES
'''

#sns.distplot(tips['total_bill'], kde = False, bins = 30)
#sns.jointplot(x = 'total_bill', y = 'tip', data = tips, kind = 'kde')
#sns.pairplot(tips, hue='sex', palette='coolwarm')

# Points with the bars to estimate the scatter
#sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std)

# Plots the numbers
#sns.countplot(x='sex', data=tips)

# Plots the points, jitter gives some random scatter for a better looking plot
#sns.stripplot(x='day', y='total_bill', data=tips, jitter=True) 

# KDE of the point distribution
#sns.violinplot(x='day', y='total_bill', data=tips, hue='smoker', split=True)
#sns.violinplot(x='day', y='total_bill', data=tips)

# Plots the points in a way that reproduces the KDE
#sns.swarmplot(x='day', y='total_bill', data=tips, color='black')

# GENERAL PLOT, you can specify the kind of plot (violin, bar, etc. using the kind= parameter)
sns.factorplot(x='day', y='total_bill', data=tips, kind='violin')

plt.show()
