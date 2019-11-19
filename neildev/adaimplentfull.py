from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.optimizers import SGD
import tensorflow as tf 
import math



num_classes = 10
batch_size = 32
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train, (60000,784))

x_test = np.reshape(x_test, (10000,784))


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# use the sgd optimizer
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

weights = np.ones(x_train.shape[0])

total = 3000
# new trainers
x_trainer = x_train[:total]
y_trainer = y_train[:total]
weights = weights[:total]


# the calc error we use in this implementation
# note that we kept the print statements for the acc, as this is 
# the correct acc. There seems to be an err that keras.fit uses
# Neil has talked about this with the prof and is a known issue

def calc_error(w_arr,model,X,y):
	err = 0
	right = 0
	wrong = 0
	# the wsum comes from the documentation. Other implements are possible and should
	# be explored
	wsum = 0
	guesses = model.predict_classes(X)
	for i in range (len(w_arr)):
		wsum += w_arr[i]
		if(guesses[i] == np.argmax(y[i])):
			right += 1
		else:
			wrong += 1
			err += 1*w_arr[i]

	# a more accurate acc
	print (" the acc is ")
	print (right / (right + wrong))
	
	return (err/wsum)


	# another version of acc not being used as of now
def calc_error2(w_arr, model, X, y):
	guesses = model.predict_classes(X)
	return np.mean(np.argmax(y, axis=1).astype(int) == guesses)


# the meat of the adaboosting.

def update_weights(w_arr,model,X,y):
	err = calc_error(w_arr,model,X,y)
	print("the error is ")
	print (err)
	alpha = math.log(((1-err)/err)) + math.log(9)
	print("the alpha is ")
	print (alpha)

	# keep the alpha value for later
	model.alpha = alpha

	length = w_arr.shape[0]


	guesses = model.predict_classes(X)

	# weight the ones we missed
	for i in range(length):

		if(guesses[i] == np.argmax(y[i])):
			pass
		else:
			w_arr[i] = w_arr[i]*math.exp(alpha)


	# normalize. This is done by hand, and if is changed to use np function,
	# make sure it gets the exact same values. This is a tricky normalization
	# due to the way weights are implemented in keras sequential models
	val = 0

	for i in range(length):
		val += (w_arr[i])*(w_arr[i])

	val = math.sqrt(val)
	w_arr /= val
	w_arr *= math.sqrt(length)

	return w_arr


# use this to make a guess. This is what is being used to evaluate, but could build
# a wrapper function to do this on a whole training set
def make_guess(models,x):

	guesses = np.zeros(10)
	for j in range(len(models)):
		#print(models[j].alpha)
		#print("the arg max is ")
		#print (np.argmax(models[j].predict(np.array([x,]))))
		for i in range(10):
			if(np.argmax(models[j].predict(np.array([x,]))) == i):
				guesses[i] += models[j].alpha
	#print( guesses)

	return np.argmax(guesses)






#number of weak learners we want to make
k = 3

models = []

for i in range(k):
	model = Sequential()

	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape:
	# here, 784-dimensional vectors.
	model.add(Dense(64, activation='relu', input_dim=img_rows*img_cols))
	model.add(Dropout(0.5))

	model.add(Dense(10, activation='softmax'))

	models.append(model)
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])


#models[0].fit(x_trainer, y_trainer,batch_size=batch_size,epochs=epochs,verbose=1,sample_weight=weights,)

#print(calc_error2(weights,models[0],x_trainer,y_trainer))
#print (weights)
#print(model.evaluate(x_trainer, y_trainer, sample_weight=weights))
#update_weights(weights,models[0],calc_error(weights,models[0],x_trainer,y_trainer))

#weights = update_weights(weights,models[0],x_trainer,y_trainer)
#print (weights)



for i in range(k):

	models[i].fit(x_trainer,y_trainer,batch_size=batch_size,verbose=1,sample_weight=weights)
	weights = update_weights(weights,models[i],x_trainer,y_trainer)
	print (models[i].alpha)



val = 0

for i in range(weights.shape[0]):
	val += (weights[i])*(weights[i])

print ("the lenghth is ")

print (math.sqrt(val))
print (math.sqrt(weights.shape[0]))

print (x_test[0].shape)


right = 0

for i in range(total):
	guess = make_guess(models,x_test[i])
	actual = np.argmax(y_test[i])

	#print(guess)
	#print(actual)

	if (guess == actual):
		right += 1

print ("the over all acc")
print (right/total)

#print (models[0].predict(np.array([x_test[0],])))

# for j in range(k):
# 	models[j].fit(x_trainer, y_trainer,

#           epochs=epochs,
#           verbose=1,
#           sample_weight=weights,
#           validation_data=(x_test, y_test)
# 	)









