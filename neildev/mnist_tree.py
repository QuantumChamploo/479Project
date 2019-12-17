# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from mnist import MNIST


mndata = MNIST("./data/")
images, labels = mndata.load_training()


train_x = images[:2000]
train_y = labels[:2000]

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.6,shuffle=True)


abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=15,
                         learning_rate=.9)

model = abc.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
#print (abc.estimators_)
print (abc.estimators_[0].score(X_test,Y_test))

