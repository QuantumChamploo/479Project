from cnnada import *

ada1 = Cnnada(4,1000)
ada1.adafit()

print("the over all acc is")
print(ada1.calc_acc())