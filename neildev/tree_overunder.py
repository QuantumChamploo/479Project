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
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt


mndata = MNIST("./data/")
images, labels = mndata.load_training()


train_x = images[:2000]
train_y = labels[:2000]

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.6,shuffle=True)

test_acc = []
overunder = []
# number of learners
kvalues = []


fig = plt.figure()
ax = plt.axes(projection='3d')

ax = plt.axes(projection='3d')

for i in range(7):
	for j in range(8):
		abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=(i+1)), n_estimators=(j+3),
                         learning_rate=.9)
		model = abc.fit(X_train, Y_train)
		test_pred = model.predict(X_test)
		train_pred = model.predict(X_train)
		high = metrics.accuracy_score(Y_train, train_pred)
		low = metrics.accuracy_score(Y_test, test_pred)
		diff = high - low
		base_acc = abc.estimators_[0].score(X_test,Y_test)
		test_acc.append(base_acc)
		overunder.append(diff)
		kvalues.append((j+3))
		print(100*(i*7 + j+1)/56)
		print("percent done")

xdata = np.array(test_acc)
ydata = np.array(kvalues)
zdata = np.array(overunder)

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

print("the data")
print("base acc")
print(test_acc)
print("number of learners")
print(kvalues)
print("overunder")
print(overunder)




plt.show()
