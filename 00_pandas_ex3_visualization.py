import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data3 = '/home/edoardo/Udemy/DataScience/Dispense/Refactored_Py_DS_ML_Bootcamp-master/07-Pandas-Built-in-Data-Viz/df3'

df3 = pd.read_csv(data3)

df3.plot.scatter(x = 'a', y = 'b')


plt.show()

