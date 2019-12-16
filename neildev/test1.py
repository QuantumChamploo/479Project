from cnnada import *

ada1 = Cnnada(9,1000)
ada1.adafit()

print("the test acc is")
print(ada1.calc_acc())

print("the train acc is ")
print (ada1.train_acc())

print("the base test acc is ")
print(ada1.base_test_acc())

print("the base train acc is ")
print(ada1.base_train_acc())