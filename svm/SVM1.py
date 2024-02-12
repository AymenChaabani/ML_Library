# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 13:04:57 2022

@author: aymen
"""
import numpy as np
from numpy import random

class SVM():
    def __init__(self,lmbd,D):
        self.lmbd = lmbd
        self.D = D + 1
        self.w = [1.]*self.D 
        
    def sign(self,x):
        if x<=0:
            return -1.
        else:
            return 1.
    def hinge_loss(self,target,y):
        return max(0,1 - target*y)
    
    def data(self,test=False):
        if test:
            with open('test.csv','r') as f:
                samples = f.readlines()

                for t,row in enumerate(samples):

                    row = row.replace('\n','')
                    row = row.split(',')

                    target = -1.

                    if row[3] == '1':
                        target = 1.
                    del row[3]

                    x = [float(c) for c in row] + [1.] #inputs + bias

                    yield t, x,target

        else:

            with open('train.csv','r') as f:
                samples = f.readlines()
                random.shuffle(samples)

                for t,row in enumerate(samples):

                    row = row.replace('\n','')
                    row = row.split(',')

                    target = -1.

                    if row[3] == '1':
                        target = 1.
                    del row[3]

                    x = [float(c) for c in row] + [1.] #inputs + bias

                    yield t, x,target
    def train(self,x,y,alpha):
        if y*self.predict(x) < 1:

            for i in range(len(x)):
                self.w[i] =  self.w[i] + alpha *( (y*x[i]) + (-2 * (self.lmbd)*self.w[i]) )                

        else:
            for i in range(len(x)):
                self.w[i] = self.w[i] + alpha * (-2 * (self.lmbd)*self.w[i])
        
        return self.w
                
    def predict(self,x):
        wTx = 0.
        for i in range(len(x)):

            wTx += self.w[i]*x[i]

        return wTx
    
    def fit(self):
        test_count = 0.
        
        tn = 0.
        tp = 0.
        
        total_positive = 0.
        total_negative = 0.

        accuracy = 0.
        loss = 0.
        
        
        last = 0
        for t, x,target in self.data(test=False):
            
            if target == last: 
                continue
            
            alpha = 1./(self.lmbd*(t+1.))
            w = self.train(x,target,alpha)
            last = target
    
        for t,x,target in self.data(test=True):
            
            pred = self.predict(x)
            loss += self.hinge_loss(target,pred)
            
            pred = self.sign(pred)
            
            
            if target == 1:
                total_positive += 1.
            else:
                total_negative += 1.
            
            if pred == target:
                accuracy += 1.
                if pred == 1:
                    tp += 1.
                else:
                    tn += 1.
            
        loss = loss / (total_positive+total_negative)
        acc = accuracy/(total_positive+total_negative)
        
        # print 'Loss', loss, '\nTrue Negatives', tn/total_negative * 100, '%', '\nTrue Positives', tp/total_positive * 100, '%','\nPrecision', accuracy/(total_positive+total_negative) * 100, '%', '\n'
    
        return loss, acc, tp/total_positive,tn/total_negative, w
                    
        