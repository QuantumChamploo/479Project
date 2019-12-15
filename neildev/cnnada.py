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





class Cnnada:
	def __init__(self,learners, size):
		self.learners = learners
		self.size = size
		self.weights = np.ones(size)
		self.x_trainer = x_train[:size]
		self.y_trainer = y_train[:size]
		self.models = []
		self.alphas = []
		self.broken = False 

		for i in range(learners):
			model = Sequential()

			# Dense(64) is a fully-connected layer with 64 hidden units.
			# in the first layer, you must specify the expected input data shape:
			# here, 784-dimensional vectors.
			model.add(Dense(64, activation='relu', input_dim=img_rows*img_cols))
			model.add(Dropout(0.5))

			model.add(Dense(10, activation='softmax'))

			self.models.append(model)
			model.compile(loss=keras.losses.categorical_crossentropy,
		              optimizer=sgd,
		              metrics=['accuracy'])




	def calc_error(self,model_num):
			w_arr = self.weights
			X = self.x_trainer
			y = self.y_trainer
			model = self.models[model_num]

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

	def adafit(self):
		for i in range(self.learners):
			self.models[i].fit(self.x_trainer,self.y_trainer,batch_size=batch_size,verbose=1,sample_weight=self.weights)
			self.weights = self.update_weights(i)		

	def update_weights(self, model_num):
		w_arr = self.weights
		model = self.models[model_num]
		X = self.x_trainer
		y = self.y_trainer
		err = self.calc_error(model_num)
		print("the error is ")
		print (err)
		alpha = math.log(((1-err)/err)) + math.log(9)
		print("the alpha is ")
		print (alpha)

		# keep the alpha value for later
		self.alphas.append(alpha)
		if(alpha < 0):
			self.broken = True

		length = self.size


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
		

	def make_guess(self,x):

		guesses = np.zeros(10)
		for j in range(len(self.models)):
			#print(models[j].alpha)
			#print("the arg max is ")
			#print (np.argmax(models[j].predict(np.array([x,]))))

			hld = np.argmax(self.models[j].predict(np.array([x,])))
			if(self.alphas[j] > 0):
				guesses[hld] += self.alphas[j]

			
			#for i in range(10):
			#	if(np.argmax(models[j].predict(np.array([x,]))) == i):
			#		if(models[j].alpha > 0):
			#			guesses[i] += models[j].alpha
			
		#print( guesses)

		return np.argmax(guesses)	



	def calc_acc(self):
		right = 0
		for i in range(10000):
			guess = self.make_guess(x_test[i])
			actual = np.argmax(y_test[i])

			if(guess == actual):
				right += 1

		return (right/10000)








