from mpl_toolkits import mplot3d

import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
from cnnada import *


fig = plt.figure()
ax = plt.axes(projection='3d')

ax = plt.axes(projection='3d')

test_acc = []
overunder = []
kvalues = []

samplesizes = np.linspace(5000,30000,6, dtype=int)
learnersizes = np.linspace(3,8,6, dtype=int)

#samplesizes = np.linspace(5000,10000,2,dtype=int)
#learnersizes = np.linspace(3,4,2,dtype=int)

print (samplesizes)
print (learnersizes)

num = 0

for i in samplesizes:
	for j in learnersizes:
		ada = Cnnada(j,i)
		ada.adafit()
		high = ada.train_acc()
		low = ada.calc_acc()
		single_acc = ada.base_test_acc()

		overunder.append((high - low))
		test_acc.append(single_acc)
		kvalues.append(j)

		num += 1

		print(100*num/(36))
		print("percent done")

xdata = np.array(test_acc)
ydata = np.array(kvalues)
zdata = np.array(overunder)

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

print("test_acc")
print(xdata)
print("num of learners")
print(ydata)
print("over under")
print(zdata)

plt.show()




