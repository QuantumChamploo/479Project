from cnnada import *
import numpy as np
import matplotlib.pyplot as plt

#ada1 = Cnnada(90,1000)
#ada1.adafit()

#if(ada1.broken):
#	print("is broken")
#	print("broke at")
#	print(ada1.brokeat)
#else:
#	print("not broken")
x = []
y = []
x_dev = []
y_dev = []

for j in range(6):
	breakvalue = []
	accvalue = []

	for i in range(15):
		ada1 = Cnnada(200,(j+1)*500)
		ada1.adafit()
		if(ada1.broken == True):
			breakvalue.append(ada1.brokeat)
			accvalue.append(ada1.base_test_acc())

	acc_arr = np.array(accvalue)
	break_arr = np.array(breakvalue)

	print("the acc arr is")
	print(acc_arr)

	x.append(np.mean(acc_arr))
	y.append(np.mean(break_arr))
	print (np.std(accvalue))
	x_dev.append(np.std(acc_arr))
	y_dev.append(np.std(break_arr))

print("printing the values")
print(x)
print(x_dev)
print(y)
print(y_dev)


plt.figure()
plt.errorbar(x, y, xerr=x_dev,yerr=y_dev, ls='none')
plt.savefig('breaking1')

plt.show()

