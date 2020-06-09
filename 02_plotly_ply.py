import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cufflinks as cf
import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import __version__
#print(__version__)

'''
    PLOTLY only works in a browser
    import plotly.express as px
    can be used to export stuff to html
'''


init_notebook_mode(connected = True)
cf.go_offline()

df = pd.DataFrame(np.random.randn(100, 4), columns = 'A B C D'.split())
#df2 = pd.DataFrame({'Category':['A', 'B', 'C'], 'Values':[32, 43 ,50]})

df.iplot()



#plt.show()
