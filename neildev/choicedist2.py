import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 

num_guessers = 5

gammas = np.linspace(0, 1, 50)

gamma = .1
n_gamma = (1-gamma)/(9)

dist = []


for i in range(10):
	if(i!=0):
		dist.append(n_gamma)
	else:
		dist.append(gamma)
print(dist)

#print(np.random.choice(10,num_guessers,p=dist))

#right = 0

right = 0

for i in range(10000):
	choice = np.array(np.random.choice(10,num_guessers,p=dist))

	bincount = np.bincount(choice)

	count = 0

	maxi = np.max(bincount)
	for j in range(len(bincount)):
		if(bincount[j] == maxi):
			count += 1

	random = np.argmax(np.bincount(choice))
	index = np.argmax(choice)
	if(random == 0):
		if(count > 1):
			right += 1/count
		else:
			right += 1
	# if(random == 5):
	# 	check = False
	# 	for j in range(num_guessers):
	# 		if(choice[j] == choice[5]):
	# 			right += .1
	# else:
	# 	right += 1

print("overall acc")
print(right/10000)
