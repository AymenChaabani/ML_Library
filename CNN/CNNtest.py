# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:01:25 2022

@author: aymen
"""

import numpy as np
import tensorflow as tf


from Layers import Linear_Layer,Convolutional_Layer,padding,reshaping,Neural_Network,pooling
from Metrics import accuracy,cross_entropy
from Activations import softmax,relu


Mnist = tf.keras.datasets.mnist

(Xtr, Ytr), (Xte, Yte) = Mnist.load_data()
X_testing = Xtr[:,:,:]
Y_testing = Ytr[:]

X_testing = X_testing/255
al = 0.3
stopper = 85.0

complete_NN = Neural_Network([
                                
                                padding(),
                                Convolutional_Layer(),
                                pooling(),
                                relu(),
                                padding(),
                                Convolutional_Layer(),
                                pooling(),
                                relu(),
                                reshaping(),
                                Linear_Layer(7*7, 24, alpha = al),
                                relu(),
                                Linear_Layer(24, 10, alpha = al),
                                softmax()

                                ])
CE = cross_entropy()

acc = accuracy()
epochs = 100
broke = 0
batches = 6000
for i in range(epochs):
    k = 0
    for ii in range(batches, 60001, batches):
        
        out = complete_NN.forward_pass(X_testing[k:ii])
        print("epoch:{} \t batch: {} \t loss: \t {}".format(i+1, int(ii/batches), CE.loss(out, Y_testing[k:ii])), end="\t")
        accur = acc.value(out, Y_testing[k:ii])*100
        print("accuracy: {}".format(accur))
        
        if accur >= stopper:
            broke = 1
            break
        complete_NN.backprop(Y_testing[k:ii])
        complete_NN.applying_sgd()
        k = ii
        
    if broke == 1:
        break
    

out = complete_NN.forward_pass(X_testing)
print("The final loss is {}".format(CE.loss(out, Y_testing)))
print("The final accuracy on train set is {}".format(acc.value(out, Y_testing)*100))
Xtest = Xte/255
out_1 = complete_NN.forward_pass(Xtest)
print("The accuracy on test set is {}".format(acc.value(out_1, Yte)*100))