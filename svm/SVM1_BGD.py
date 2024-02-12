# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 11:42:50 2022

@author: aymen
"""

import numpy as np
 
class SVM():
    def sign(self,x):
        if x<=0:
            return -1.
        else:
            return 1.
    def grad_w(self,x, y, w, b, C):
        mask = y * (x.dot(w) + b) < 1
        dL = - x * y.copy().reshape((-1, 1))
        mask = mask.reshape((-1, 1)) * np.ones(dL.shape)
        dL = dL * mask

        return w + C * np.sum(dL, axis=0)

    def grad_b(self,x, y, w, b, C):
        mask = y * (x.dot(w) + b) < 1
        dL = - y.copy().reshape((-1, 1))
        mask = mask.reshape((-1, 1)) * np.ones(dL.shape)
        dL = dL * mask
        
        return C * np.sum(dL, axis=0)

        
    def hinge_loss(self,target,y):
        return max(0,1 - target*y)
    
    def __init__(self,lmbd=1,D=3,epochs=100,batch_size=32,learning_rate=1e-5):

        self.lmbd = lmbd
        self.D = D
        self.w = [1.]*self.D
        self.b = 1
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        pass
    def learn (self,data):
        feature_len = len(data)
        self.batch_num = feature_len//self.batch_size

        for epoch in range(self.epochs):
            print("epoch n:", epoch+1)
            np.random.seed(epoch)
            np.random.shuffle(data)
            for i in range(self.batch_num):
                features_batch = data[i*self.batch_size:(i+1)*self.batch_size,:-1]
                targets_batch = data[i*self.batch_size:(i+1)*self.batch_size,-1]
                self.w=self.w- self.lr*self.grad_w(features_batch,targets_batch,self.w,self.b,100)
                self.b=self.b - self.lr*self.grad_b(features_batch,targets_batch,self.w,self.b,100)
                print("w:{},b:{}".format(self.w,self.b))
    def infer(self,x):
        wTx = 0.
        for i in range(len(x)):

            wTx += self.w[i]*x[i]
        wTx+=self.b
        return wTx
    