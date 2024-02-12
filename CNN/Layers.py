# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 09:37:47 2022

@author: aymen
"""

import numpy as np

class Linear_Layer:

    def __init__(self, d_in, d_out, alpha = 0.01, W = None, bias = None, Norm = 1):
        self.alpha = alpha
        if W == None:
            self.W = np.random.randn(d_in, d_out)/Norm

        else:
            self.W = W

        if bias == None:
            self.bias = np.random.randn(d_out)/Norm

        else:
            self.bias = bias
        

    def forward_pass(self, Input):
        self.Input = Input
        self.Output = np.matmul(Input, self.W) + self.bias
        return self.Output

    
    def backprop(self, grad_previous):
        Num_records= self.Input.shape[0]
        self.grad = np.matmul((self.Input.transpose()), grad_previous)/Num_records
        self.grad_bias = (grad_previous.sum(axis=0))/Num_records
        self.grad_a = np.matmul(grad_previous, self.W.transpose())
        return self.grad_a



    def applying_sgd(self):
            self.W = self.W - (self.alpha*self.grad)
            self.bias = self.bias - (self.alpha*self.grad_bias)


class padding():
    
    def __init__(self, pad = 1):
        self.pad = pad

    def forward_pass(self, Input):
        Output = np.pad(Input , ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),'constant', constant_values=0)
        return Output

    def backprop(self, Output):
        return (Output[:, 1:(Output.shape[1]-1),1:(Output.shape[2]-1)])

    def applying_sgd(self):
        #nothing to be done
        pass

class Convolutional_Layer:
    def __init__(self, filter_dim = 3, stride = 1, pad = 1, alpha=0.01):
        self.filter_dim = filter_dim
        self.stride = stride
        self.filter = np.random.randn(self.filter_dim, self.filter_dim)
        self.filter = self.filter/self.filter.sum()
        self.bias = np.random.rand()/10
        self.pad = pad
        self.alpha = alpha

    def convolving(self, X, fil, dimen_x, dimen_y):
        z = np.zeros((dimen_x, dimen_y))
        for i in range(dimen_x):
            for ii in range(dimen_y):
                temp = np.multiply(X[i : i+fil.shape[0], ii : ii+fil.shape[1]], fil)
                z[i,ii] = temp.sum()
        return z
        
        
    def forward_pass(self, X):
        self.X = X
        (d, p, t) = self.X.shape
        dimen_x = int(((p - self.filter_dim)/self.stride) + 1)
        dimen_y = int(((t - self.filter_dim)/self.stride) + 1)
        self.z = np.zeros((d, dimen_x, dimen_y))
        for i in range(d):
            self.z[i] = (self.convolving(self.X[i], self.filter, dimen_x, dimen_y) + self.bias)

        return self.z

    def backprop(self, grad_z):
        (d, p, t) = grad_z.shape
        filter_1 = np.flip((np.flip(self.filter, axis = 0)), axis = 1)
        self.grads = np.zeros((d, p, t))
        for i in range(d):
            self.grads[i] = self.convolving(np.pad(grad_z[i], ((1,1), (1,1)), 'constant', constant_values = 0), filter_1, p, t)

        self.grads = np.pad(self.grads, ((0,0),(1,1),(1,1)), 'constant', constant_values = 0)

        self.grad_filter = np.zeros((self.filter_dim, self.filter_dim))

        for i in range(self.filter_dim):
            for ii in range(self.filter_dim):
                self.grad_filter[i, ii] = (np.multiply(grad_z, self.X[:, i:p+i, ii:t+ii])).sum()
        self.grad_filter = self.grad_filter/(d)

        self.grad_bias = (grad_z.sum())/(d)
        return self.grads

    def applying_sgd(self):
        self.filter = self.filter - (self.alpha*self.grad_filter)
        self.bias = self.bias - (self.alpha*self.grad_bias)


class pooling:

    def __init__(self, pool_dim = 2, stride = 2):
        self.pool_dim = pool_dim
        self.stride = stride

    def forward_pass(self, Input):
        (q, p, t) = Input.shape
        Output_x = int((p - self.pool_dim) / self.stride) + 1
        Output_y = int((t - self.pool_dim) / self.stride) + 1
        after_pool = np.zeros((q, Output_x, Output_y))
        for ii in range(0, q):
            liss = []
            for i in range(0,p,self.stride):
                for j in range(0,t,self.stride):
                    if (i+self.pool_dim <= p) and (j+self.pool_dim <= t):
                        temp = Input[ii, i:(i+(self.pool_dim)), j:(j+(self.pool_dim))]
                        temp_1 = np.max(temp)
                        liss.append(temp_1)
            liss = np.asarray(liss)
            liss = liss.reshape((Output_x, Output_y))
            after_pool[ii] = liss
            del liss
        return after_pool

    def backprop(self, Output):
        (a,b,c) = Output.shape   
        cheated = np.zeros((a,2*b,2*c))
        for k in range(0, a):
            pooled_transpose_re = Output[k].reshape((b*c))
            count = 0
            for i in range(0, 2*b, self.stride):
                for j in range(0, 2*c, self.stride):
                    cheated[k, i:(i+(self.stride)),j:(j+(self.stride))] = pooled_transpose_re[count]
                    count = count+1
        return cheated

    def applying_sgd(self):
        #nothing to be done here
        pass


class Neural_Network:

    def __init__(self, Network): #input list of the object layers
        self.Network = Network

    def forward_pass(self, X):
        n = X
        for i in self.Network:
            n = i.forward_pass(n)
            
            
        return n
    
    def backprop(self, Y):
        m = Y
        for i in (reversed(self.Network)):
            m = i.backprop(m)

    def applying_sgd(self):
        for i in self.Network:
            i.applying_sgd()


class reshaping:
    #This class does the 3D to 2d reshaping and vice versa
    def __init__(self):
        pass

    def forward_pass(self, Input):
        self.shape_Input = Input.shape
        
        self.final_Input = Input.reshape(self.shape_Input[0], self.shape_Input[1]*self.shape_Input[2])
        return self.final_Input
    
    def backprop(self, Output):
        return (Output.reshape(self.shape_Input[0], self.shape_Input[1], self.shape_Input[2]))

    def applying_sgd(self):
        #nothing to be done here
        pass

