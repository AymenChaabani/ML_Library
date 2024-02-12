# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 09:44:31 2022

@author: aymen
"""

import numpy as np
class cross_entropy:

    def __init__(self):
        pass
    
    def expansion(self, t):
        (a,) = t.shape
        Y = np.zeros((a,10))
        for i in range(0,a):
            Y[i,t[i]] = 1
        return Y

    def loss(self, A, Y):
        exp_Y = self.expansion(Y)
        (u,i) = A.shape
        loss_matrix = np.zeros((u,i))
        for j in range(u):
            for jj in range(i):
                if exp_Y[j,jj] == 0:
                    loss_matrix[j,jj] = np.log(1 - A[j,jj])
                else:
                    loss_matrix[j,jj] = np.log(A[j,jj])
        

        return ((-(loss_matrix.sum()))/u)

class accuracy:
    def __init__(self):
        pass

    def value(self, out, Y):
        self.out = np.argmax(out, axis=1)
        p = self.out.shape[0]
        total = 0
        for i in range(p):
            if Y[i]==self.out[i]:
                total += 1
        return total/p