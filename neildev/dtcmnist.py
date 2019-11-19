from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import keras
from keras.datasets import mnist
from sklearn import metrics


# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train, (60000,784))

x_test = np.reshape(x_test, (10000,784))


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

numexam = 5

x_train = x_train[:numexam]
y_train = y_train[:numexam]

max_depth = 1
n_estimators = 5


# Create adaboost classifer object
abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators,
                         learning_rate=.9)

model = abc.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print ("acc of first estimator")
#print (abc.estimators_)
for i in range(n_estimators):
	print("the acc of the %i th estimator is : ", i +1)

	print (abc.estimators_[i].score(x_test,y_test))