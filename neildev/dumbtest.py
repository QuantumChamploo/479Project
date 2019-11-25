import numpy as np

arr = np.ones(1000)

print (arr)

arr = arr/np.linalg.norm(arr)
print (arr)

val = 0

for i in range(1000):
	val += arr[i]

print (val)