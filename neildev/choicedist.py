import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 

num_guessers = 5

gamma = .1

n_gamma = (1-gamma)/(9)

def guess(choices, dist):
	guess = np.random.choice(choices, p=dist)
	return guess

choices = np.arange(0, 10)
dist = np.ones(len(choices))*(1./len(choices))
#dist[0] = 0.1
dist = dist/np.sum(dist)
print(dist)
guesses = np.asarray([guess(choices, dist) for i in range(5)])
print('number of times i got 0: {}'.format(len(guesses[guesses==0])))
plt.hist(guesses)
plt.savefig('this_fig')
#for i in range(10):
#	if(i!=5):
#		dist.append(n_gamma)
#	else:
#		dist.append(gamma)
#print(dist)

#print(np.random.choice(10,num_guessers,p=dist))

#right = 0


for i in range(10000):
	choice = np.array(np.random.choice(10,num_guessers,p=dist))
	print("choice vec")
	print(choice)

	random = np.argmax(np.bincount(choice))
	index = np.argmax(choice)
	if(random == 5):
		check = False
		for j in range(num_guessers):
			if(choice[j] == choice[5]):
				right += .1
	else:
		right += 1

print("overall acc")
print(right/10000)
