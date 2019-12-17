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



acc = []
breaks = []
for i in range(5):
	X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.6,shuffle=True)
	abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=200,
	                         learning_rate=.9)

	model = abc.fit(X_train, Y_train)

	y_pred = model.predict(X_test)

	print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
	#print (abc.estimators_)
	for i in range(200):
		hld = abc.estimators_[i].score(X_test,Y_test)
		print(i)
		print(hld)
		if(hld<.1):
			acc.append(abc.estimators_[0].score(X_test,Y_test))
			breaks.append(i+1)
			break

print(acc)
print(breaks)
acc_arr = np.array(acc)
break_arr = np.array(breaks)
print("avg acc is")
print(np.mean(acc_arr))
print("dev is ")
print(np.std(acc_arr))
print("avg breaking point")
print(np.mean(break_arr))
print("with dev")
print(np.std(break_arr))


