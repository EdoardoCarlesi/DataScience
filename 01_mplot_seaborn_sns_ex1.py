import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style('whitegrid')

titanic = sns.load_dataset('titanic')

head = titanic.head()

print(head.columns)

# Joint plot
#sns.jointplot(y='age', x='fare', data = titanic)

# Dist plot
#sns.distplot(titanic['fare'], kde = False, color = 'red')

# Bar plot
#sns.barplot(x='class', y='age', data = titanic)

# Box plot
#sns.boxplot(x='class', y='age', data = titanic)

# Violin plot
#sns.violinplot(x='class', y='age', data = titanic)

# Swarmplot
#sns.swarmplot(x='class', y='age', data = titanic)

# Heat map
sns.heatmap(titanic.corr(),cmap='coolwarm')

# Separate by column
#g = sns.FacetGrid(data=titanic,col='sex')
#g.map(plt.hist,'age')

plt.show()
