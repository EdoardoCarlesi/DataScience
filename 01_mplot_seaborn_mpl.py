import matplotlib.pyplot as plt
import numpy as np

import tkinter

# NOTEBOOK ONLY
#%matplotlib inline


x = np.linspace(0, 5, 11)
y = x ** 2

'''
#plt.xlabel('X')
#plt.ylabel('Y')
#print(y)
#plt.plot(x, y, 'r-')

plt.subplot(1, 2, 1)
plt.plot(x, y, 'r-')

plt.subplot(1, 2, 2)
plt.plot(y, x, 'b-')

plt.show()
'''

# Object oriented method

'''
fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.set_xlabel('X')
axes.set_title('Title')

axes.plot(x, y)

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])

axes1.plot(x, y)
axes2.plot(y, x)
'''

#fig, axes = plt.subplots(nrows = 1, ncols = 2)
#for ax in axes:
#    ax.plot(x, y)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3,2), dpi=150)

axes[0].plot(x, y)
axes[1].plot(x, y)

plt.tight_layout()
plt.show()


