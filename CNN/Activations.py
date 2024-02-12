# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 09:37:54 2022

@author: aymen
"""


import numpy as np

class softmax:

    def __init__(self):
        pass
    
    def expansion(self, t):
        (a,) = t.shape
        Y = np.zeros((a,10))
        for i in range(0,a):
            Y[i,t[i]] = 1
        return Y
    
    def forward_pass(self, Input):
        self.Input =  Input
        (p,t) = self.Input.shape
        self.a = np.zeros((p,t))
        for line in range(0,p):
            for column in range(0,t):
                self.a[line,column] = (np.exp(self.Input[line,column]))/(np.sum(np.exp(self.Input[line,:])))
        return self.a

    def backprop(self, Output):
        y = self.expansion(Output)
        self.grad = (self.a - y)
        return self.grad

    def applying_sgd(self):
        #nothing to be done
        pass

class relu:
    def __init__(self):
        pass

    def forward_pass(self,Input ):
        
        if (len(Input.shape) == 3):

            Input_temp = Input.reshape((Input.shape[0], Input.shape[1]*Input.shape[2]))
            Input_temp_1 = self.forward_pass(Input_temp)
            self.Output = Input_temp_1.reshape((Input.shape[0], Input.shape[1], Input.shape[2]))
            return (self.Output)

        else:
            (p,t) = Input.shape
            self.Output1 = np.zeros((p,t))
            for i in range(0,p):
                for ii in range(0,t):
                        self.Output1[i,ii] = max([0,Input[i,ii]])
            return self.Output1

    def derivative(self, Input):
        if Input>0:
            return 1
        else:
            return 0
    
    def backprop(self, grad_previous):
        
        if (len(grad_previous.shape)==3):

            (d, p, t) = grad_previous.shape
            self.grad = np.zeros((d, p, t))
            
            for i in range(d):
                for ii in range(p):
                    for iii in range(t):
                        self.grad[i, ii, iii] = (grad_previous[i, ii, iii] * self.derivative(self.Output[i, ii, iii]))
            
            return (self.grad)

        else:
            (p,t) = grad_previous.shape
            self.grad = np.zeros((p,t))
            for i in range(p):
                for ii in range(t):
                    self.grad[i,ii] = grad_previous[i,ii] * self.derivative(self.Output1[i,ii])
            return (self.grad)

    
    def applying_sgd(self):
        #nothing to be done
        pass
