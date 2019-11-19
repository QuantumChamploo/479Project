pytho# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier



# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

numexam = 150

X = X[:numexam]
y = y[:numexam]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

print ("printing shape")
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
print (y_train)
# Create adaboost classifer object
abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=5,
                         learning_rate=.9)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print (abc.estimators_)
print (abc.estimators_[0].score(X_test,y_test))

