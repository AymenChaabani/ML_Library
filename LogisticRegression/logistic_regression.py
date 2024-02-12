# -*- coding: utf-8 -*-
"""
Created on Sun May 15 10:19:44 2022

@author: aymen
"""
import numpy as np
#####################################################################
class LogisticRegression:

    def __init__(self,learningRate=1,num_batch=1,num_epoch=1):
        self.learningRate=learningRate
        self.num_batch=num_batch
        self.num_epoch=num_epoch
        self.w_best=0
    def sigmoid(self,x):
        #this function returs the sigmoid(x)
        return 1 / (1 + np.exp(-x))
    def compute_loss(self,X, y, theta, intercept=0, epsilon: float = 1e-5):
    # Loss(w) = -(y_i*log(h_w(x_i))) + (1-y_i)*log(1-h_w(x_i))
    # theta are our parameters w of the Logistic Regression
    # 
        batch_size = len(y)
        h = self.sigmoid(np.dot(X,theta)+intercept)
        loss = (1/batch_size)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
        return loss
    
    def learn(self,features,targets):
        train=np.concatenate((np.ones((features.shape[0],1)), features),axis=1) #add the bias to the given features input
        w0=np.ones((train.shape[1],1)) #initilisation of w
        print(w0.shape)
        self.batchsize=train.shape[0]//self.num_batch #determine the batch size given the wanted number of batches 
        for epoch in range(self.num_epoch):
            print("epoch number {} :\n".format(epoch+1))
            for batch in range(self.num_batch):
                begin= batch*self.batchsize
                end=(batch+1)*self.batchsize
                train_batch= train[begin:end]
                targets_batch= np.array(targets[begin:end]).reshape(-1,1)
                y_predicted=self.sigmoid(np.dot(train_batch,w0))
                loss=self.compute_loss(train_batch,targets_batch,w0)
                print(loss)
                gradient = -(2./ self.batchsize)*np.dot(train_batch.T,(y_predicted - targets_batch))
                w0-=gradient*self.learningRate
        self.w0=w0[0]
        self.w=w0[1:]
    def infer(self,features):
        #predict the eventual label given the features input and the calculated w and bias w0
        y_predicted= self.sigmoid(features.dot(self.w)+self.w0)
        return y_predicted
#this is another model variant that differentiate from the first model in the gradient descent, 
# that here the derivation referencing w and the bias is being done separately
class LogisticRegression1:

    def __init__(self,learningRate=1,num_batch=1,num_epoch=1):
        self.learningRate=learningRate
        self.num_batch=num_batch
        self.num_epoch=num_epoch
        self.w_best=0
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def compute_loss(self,X, y, theta, intercept=0, epsilon: float = 1e-5):

        batch_size = len(y)
        h = self.sigmoid(np.dot(X,theta)+intercept)
        loss = (1/batch_size)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
        return loss
    
    def learn(self,features,targets):
        targets=targets.reshape(-1,1)
        w0=np.ones((features.shape[1],1))
        bias=0
        for epoch in range(self.num_epoch):
            print("epoch number {} :\n".format(epoch+1))
            y_predicted=self.sigmoid(np.dot(features,w0))
            loss=self.compute_loss(features,targets,w0)
            print(loss)
            dw = -(2./ features.shape[1])*np.dot(features.T,(y_predicted - targets))
            db = (1 / features.shape[1]) * np.sum(y_predicted - targets)
            w0-=dw*self.learningRate
            bias-=db*self.learningRate
        self.w0=bias
        self.w=w0
    def infer(self,features):
        y_predicted= self.sigmoid(features.dot(self.w)+self.w0)
        return y_predicted
