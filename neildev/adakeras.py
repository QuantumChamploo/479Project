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
from sklearn.ensemble import AdaBoostRegressor
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split

# What I believe is a proper work around to make ada work well with CNN

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

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

x_train = x_train[:1000]
y_train = y_train[:1000]




def simple_model():                                           
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2, input_shape=(x_train.shape[1],)))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['accuracy'])
    return model

ann_estimator = KerasRegressor(build_fn= simple_model, epochs=1, batch_size=10, verbose=1)

boosted_ann = AdaBoostRegressor(base_estimator= ann_estimator, n_estimators=5)
boosted_ann.fit(x_train, y_train)
score = boosted_ann.score(x_test, y_test)
print (score)
print (type(boosted_ann.estimators_[0]))

prediction = boosted_ann.estimators_[0].score(x_test, y_test)
print (prediction)
