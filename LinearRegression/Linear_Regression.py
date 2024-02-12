# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:26:37 2022

@author: aymen
"""
#Importation 
import numpy as np
###########################################################"
#Model 1 : This model uses the Linear equation to find the w_best and to do the prediction
class LinearRegressionLinearEquation:
    def __init__(self):
        #initialisation of w_best to 0
        self.w_best=0
    def Matrixinver(self, Matrix):
        #This function returns the inverser of the matrix given as input
        return np.linalg.inv(Matrix)
    def learn(self, features: np.array, targets: np.array):


        train=np.concatenate((np.ones((features.shape[0],1)), features.to_numpy()),axis=1)  #add bias to the input features matrix
        
        self.w_best=np.dot(np.dot(self.Matrixinver(np.dot(train.T,train)),train.T),targets) #predict the w_best based on the direct linear equation
        
    def infer(self, features):
        
        train=np.concatenate((np.ones((features.shape[0],1)), features.to_numpy()),axis=1)  
        y_predicted=train.dot(self.w_best)
        
        return y_predicted

    def rmse(X_1, X_2):
        #this function return the root mean square error given two inputs X_1 and X_2
        return np.sqrt(np.sum((X_1-X_2)**2))
#Model 2 : This model uses the gradient descent method to train ,to find the w_best and to do the prediction
class LinearRegressiongradient(LinearRegressionLinearEquation):
    def __init__(self,learningRate=1,num_batch=1,num_epoch=1):
        super(LinearRegressiongradient, self).__init__()
        self.epochs=num_epoch
        self.num_batch=num_batch
        self.learningRate=learningRate
        
    def learn(self, features, targets):
        train=np.concatenate((np.ones((features.shape[0],1)), features.to_numpy()),axis=1) #add bias to the input features matrix
        self.batchsize=train.shape[0]//self.num_batch
        print("batch size: ",self.batchsize)
        w0=np.zeros((train.shape[1],1))
        print("w0 ",w0)

        for epoch in range(self.epochs):
            print("epoch number {} :\n".format(epoch+1))
            #this part of the code is used to perform the gradient descent on a batch level with a given size in place of all the input data
            #for batch in range(self.num_batch):
                
            #    begin= batch*self.batchsize
            #    end=(batch+1)*self.batchsize
                
            #    train_batch= train[begin:end]
            #    targets_batch= np.array(targets[begin:end]).reshape(-1,1)
            y_predicted=train.dot(w0)
            error= targets-y_predicted
            print("error ",error)
            mse=np.mean(error ** 2.)
            print("mse : ", mse)
            gradient = -(2./ train.shape[1])*np.dot(train.T,error)
            w0-=gradient*self.learningRate
        self.wi=w0[0] #bias
        self.w=w0[1:] # w_best
        print("W0: \n {} \n w:\n {}".format(self.wi,self.w))
    def infer(self,features):
        y_predicted= features.dot(self.w)+self.wi
        return y_predicted
