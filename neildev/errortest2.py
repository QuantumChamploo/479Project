import numpy as np
import matplotlib.pyplot as plt
# example data
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)

plt.figure()

plt.errorbar(x, y, xerr=[.1,.3,.2,.4,.1,.3,.1,.2])

plt.show()