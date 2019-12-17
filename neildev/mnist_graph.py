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
import matplotlib
import matplotlib.pyplot as plt


mndata = MNIST("./data/")
images, labels = mndata.load_training()


train_x = images[:20000]
train_y = labels[:20000]

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.6,shuffle=True)

x = []
y = []
#was 15 for smallers sets
for i in range(10):
	abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=(i+1)), n_estimators=64,learning_rate=.9)
	model = abc.fit(X_train, Y_train)
	y_pred = model.predict(X_test)
	x.append(abc.estimators_[0].score(X_test,Y_test))
	y.append(metrics.accuracy_score(Y_test, y_pred))
	print(100*(i/15))
	print("percent done")

plt.plot(x,y,marker='o')

plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("base learner rate",fontsize=18)
plt.ylabel("ensemble rate",fontsize=18)
plt.title("64 learners")
print("values")
print("base accs")
print(x)
print("overalll acc")
print(y)

plt.show()