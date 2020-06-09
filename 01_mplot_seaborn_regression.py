import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')

tips.head()

# Test multiple plots, aspect and ratios
#sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', markers=['o', 'v'], scatter_kws={'s':100})

# Aspect is the ration length / height
#sns.lmplot(x='total_bill', y='tip', data=tips, col='day', hue='sex', aspect=0.6, size=8)


# Test style and size

'''
plt.figure(figsize=(12,3))

sns.set_context('notebook', font_scale=3)

# Change background
sns.set_style('whitegrid')
sns.countplot(x='sex', data=tips)

# Removes ticks on the size
sns.despine(left=True, bottom=True)
'''

sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', palette='coolwarm')


plt.show()

