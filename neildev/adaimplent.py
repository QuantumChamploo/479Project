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


# new trainers
x_trainer = x_train[:1000]
y_trainer = y_train[:1000]
weights = weights[:1000]

# def reg_acc(y_true, y_pred):
# 	right = 0
# 	wrong = 0
# 	print ("y true type")
# 	print (type(y_true))
# 	print (tf.size(y_true))

# 	for i in range(9):
# 		if(y_true[i] == y_pred[i]):
# 			right += 1
# 	else:
# 		wrong += 1
# 	print (right / (right + wrong))


def calc_error(w_arr,model,X,y):
	err = 0
	right = 0
	wrong = 0
	guesses = model.predict_classes(X)
	for i in range (len(w_arr)):
		if(guesses[i] == np.argmax(y[i])):
			right += 1
		else:
			wrong += 1
			err += 1

	print (" the acc is ")
	print (right / (right + wrong))
	
	return (err/len(w_arr))






def update_weights(w_arr,model,err):
	alpha = .5 * math.log((1-err)/err)
	print (alpha)
	for val in w_arr:
		return



#weights /= (x_trainer.shape[0])

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


models[0].fit(x_trainer, y_trainer,
		  batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          sample_weight=weights,

)


update_weights(weights,models[0],calc_error(weights,models[0],x_trainer,y_trainer))

# for j in range(k):
# 	models[j].fit(x_trainer, y_trainer,

#           epochs=epochs,
#           verbose=1,
#           sample_weight=weights,
#           validation_data=(x_test, y_test)
# 	)









