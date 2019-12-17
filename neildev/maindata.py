from cnnada import *
import numpy as np
import matplotlib.pyplot as plt


x = []
y = []
z = []

x_dev = []
z_dev = []

num = 0
def run_trial(num, learners, size):
	testacc = []
	baseacc = []

	for i in range(3):
		ada1 = Cnnada(learners,size)
		num += 1
		ada1.adafit()
		hld = ada1.base_test_acc()
		hld2 = ada1.calc_acc()
		testacc.append(hld2)
		baseacc.append(hld)

	return testacc, baseacc

test, base = run_trial(num,5,1000)

test_acc_arr = np.array(test)
base_acc_arr = np.array(base)

learners = 5


x.append(np.mean(base_acc_arr))
y.append(learners)
z.append(np.mean(test_acc_arr))

x_dev.append(np.std(base_acc_arr))
z_dev.append(np.std(test_acc_arr))

plt.figure()
plt.errorbar(x,z,xerr=x_dev,yerr=z_dev,ls = 'none')

print("values")
print(x)
print(z)

plt.show()
