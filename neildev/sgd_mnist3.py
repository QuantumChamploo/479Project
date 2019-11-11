
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.optimizers import SGD
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

num_classes = 10
batch_size = 32
epochs = 1

# the CNN does not work by default with the adaclassifier. Known issue
# fix is in adakeras.py

# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print ("x shape")
print(x_train.shape)
x_train = np.reshape(x_train, (60000,784))
print ("x shape")
print(x_train.shape)
print (x_train[0])

print ("y shape")
print(y_train.shape)
print(y_test)

print ("xtest shape")
print(x_test.shape)

x_test = np.reshape(x_test, (10000,784))







x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()

# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 784-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=img_rows*img_cols))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))





# use the sgd optimizer
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

#weight array
weights = np.ones(x_train.shape[0])


# model compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])




classifier = AdaBoostClassifier(base_estimator=model,
    n_estimators=10,
    learning_rate=.0000000001
)


# # new trainers
x_trainer = x_train[:10000]
y_trainer = y_train[:10000]
# weights = weights[:10000]

# print ("train shape is ")
# print (x_trainer.shape[0])

# weights /= (x_trainer.shape[0])

# #fit the model 
print(y_trainer.shape)
print (y_trainer.shape)
print (x_trainer.shape)
print (y_trainer[1])

classifier.fit(x_trainer, y_trainer)
print ('something')

print (classifier.estimators_)

print ("something else")



# model2.fit(x_trainer, y_trainer,

#           epochs=epochs,
#           verbose=0,
#           sample_weight=weights,
#           validation_data=(x_test, y_test)
# )


# score1 = model.evaluate(x_test, y_test, verbose=0)
# score2 = model.evaluate(x_test, y_test, verbose=0)


# print('Test1 loss:', score1[0])
# print('Test1 accuracy:', score1[1])

# print('Test2 loss:', score2[0])
# print('Test2 accuracy:', score2[1])






