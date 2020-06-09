import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = '/home/edoardo/Udemy/DataScience/Dispense/Refactored_Py_DS_ML_Bootcamp-master/07-Pandas-Built-in-Data-Viz/df1'
data2 = '/home/edoardo/Udemy/DataScience/Dispense/Refactored_Py_DS_ML_Bootcamp-master/07-Pandas-Built-in-Data-Viz/df2'
data3 = '/home/edoardo/Udemy/DataScience/Dispense/Refactored_Py_DS_ML_Bootcamp-master/07-Pandas-Built-in-Data-Viz/df3'

df1 = pd.read_csv(data1)
df2 = pd.read_csv(data2)
df3 = pd.read_csv(data3)

# HISTOGRAM PLOTS

#df1['A'].hist(bins=30)

#df1['A'].plot(kind='hist', bins=30)

#df1['A'].plot.hist(bins=30)

#df2.plot.area(alpha=0.6)
#df2.plot.bar()
#df2.plot.bar(stacked = True)

# LINE DOES NOT WORK porcodddio
#df1.plot.line(x=df1.index, y='B')

# SCATTERPLOTS
#df1.plot.scatter(x='A', y='B', c = 'C', cmap='coolwarm')
#df1.plot.scatter(x='A', y='B', s = df1['C'] * 50)

# Box plot
#df2.plot.box()

# Hexagon heat map
#df = pd.DataFrame(np.random.randn(1000,2), columns = ['a', 'b'])
#df.plot.hexbin(x = 'a', y = 'b', gridsize = 30)

# Kernel density estimation
df2['a'].plot.density()


plt.show()
