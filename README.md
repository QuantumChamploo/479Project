The outline of project implementation

In this project we attempt to understand how weak is a weak learner(or weak model). Here
 out implementation is a Fully Connected Neural Net, with a hand made ADA boosting 
 algorithm. This can be found in the adadimplementfull.py. We will be controlling the 
 strength of the learner by changing hyperparameters. The easiest of these is the number
  of training examples. Other hyperparameters that could be changed is the learning rate 
  of the SGD optimizer. Both of these can be explored in data analysis of the project. 
  Graphs should be made by varying these hyper parameters and plotting over all accuracy. 

  The current implementation makes a set of models, with built functions to make these models using ADA and taking guess. It then prints the accuracy of the set of models. If
   you have question about ada boosting , please check the paper in the docs folder. Some wrapper functions can be made make the data gathering process easier. If you are 
   feeling
   particularly ambitios, this could be made into a class and use object oriented programming
   to do this. I will probably do this by the time of the final paper. It shouldnt be too 
   hard and be a good coding excercise. 


   It may seem unnecessary to have hand made functions to calculate accuracy, but keras 
   as a very weird definition of accuracy, so when doing formal analysis we need to use o
   one we understand. Keep this in mind when expanding on the code.

   An important piece of our analysis will be comparing our results to combinatorial 
   probability of a non weighted learner. For more information on this look to lecture 7
   slides 13-15. because we are not a binary classifier, and I willl spend some time
    modifying it to our needs, but understand what this and why its important

   Hopefully some good results come out

