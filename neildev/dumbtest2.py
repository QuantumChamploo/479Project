import numpy as np
import math

arr = np.ones(1000)

val = 0

for i in range(1000):
	val += (arr[i])*(arr[i])

arr /= val

print(arr)
 
total = 0
for i in range(1000):
	total += arr[i]

#val = math.sqrt(val)
print (val)
print (total)