# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:52:57 2022

@author: aymen
"""

import numpy as np

class SIMPLEANN():
    def calculate_loss(self,X,y,model):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation to calculate our predictions
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(len(X)), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./len(X) * data_loss
    def sigmoid(self,X):
        return 1/(1+np.exp(- X ))
    def ReLu(self,X):
        return np.maximum(0,X)
    def derivation_ReLu(self,X):
        y=np.copy(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]) :
                if X[i,j] != 0 : y[i,j]=1
        return y
    def __init__(self,input_dim,output_dim,hidden_layer_dim,activation_function,epochs,epsilon=0.01,reg_lamba=0.01):
           
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.epsilon = epsilon
        self.reg_lambda = reg_lamba
        self.epochs = epochs
        self.activation_function = activation_function
    
    def learn(self,features,labels,print_info=False,annealling=False):
        np.random.seed(0)
        W1 = np.random.randn(self.input_dim, self.hidden_layer_dim) / np.sqrt(self.input_dim)
        b1 = np.zeros((1, self.hidden_layer_dim))
        W2 = np.random.randn(self.hidden_layer_dim, self.output_dim) / np.sqrt(self.hidden_layer_dim)
        b2 = np.zeros((1, self.output_dim))
        self.model={}
        for epoch in range(self.epochs):
             print("epoch : ", epoch+1)
             # Forward propagation
             z1 = features.dot(W1) + b1
             if self.activation_function == 1 :
                 a1 = np.tanh(z1)
             elif self.activation_function == 2 :
                 a1 = self.sigmoid(z1)
             else :
                 a1 = self.ReLu(z1)
             z2 = a1.dot(W2) + b2
             exp_scores = np.exp(z2)
             probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
             delta3 = probs
             delta3[range(len(features)), labels] -= 1
             dW2 = (a1.T).dot(delta3)
             db2 = np.sum(delta3, axis=0, keepdims=True)
             if self.activation_function == 1 :
                 delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
             elif self.activation_function == 2 :
                 delta2 = delta3.dot(W2.T) * np.multiply(a1, (1-a1))
             else :
                 delta2 = delta3.dot(W2.T) * self.derivation_ReLu(a1)
             delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
             dW1 = np.dot(features.T, delta2)
             db1 = np.sum(delta2, axis=0)
             
             # Add regularization terms (b1 and b2 don't have regularization terms)
             dW2 += self.reg_lambda* W2
             dW1 += self.reg_lambda* W1
             
             # Gradient descent parameter update
             W1 += -self.epsilon * dW1
             b1 += -self.epsilon * db1
             W2 += -self.epsilon * dW2
             b2 += -self.epsilon * db2
             #annealing the epsilon
             if annealling == True:
                 self.epsilon = self.epsilon*np.exp( - epoch)
        
             # Assign new parameters to the model
             self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
             # Optionally print the loss.
             # This is expensive because it uses the whole dataset, so we don't want to do it too often.
             if print_info and epoch % 1000 == 0:
                print("Loss after iteration %i: %f" %(epoch, self.calculate_loss(features,labels,self.model)))
            
    def infer(self,feature):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # Forward propagation
        z1 = feature.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
#simple Ann wih Batch gradient descent
class SIMPLEANNMINIBATCH():
    def calculate_loss(self,X,y,model):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation to calculate our predictions
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(len(X)), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./len(X) * data_loss
    def sigmoid(self,X):
        return 1/(1+np.exp(- X ))
    def ReLu(self,X):
        return np.maximum(0,X)
    def derivation_ReLu(self,X):
        y=np.copy(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]) :
                if X[i,j] != 0 : y[i,j]=1
        return y
    def __init__(self,input_dim,output_dim,hidden_layer_dim,activation_function,batch_size,epochs,epsilon=0.01,reg_lamba=0.01):
           
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.epsilon = epsilon
        self.reg_lambda = reg_lamba
        self.epochs = epochs
        self.activation_function = activation_function
        self.batch_size = batch_size
    
    def learn(self,features,labels,print_info=False,annealling=False):
        np.random.seed(0)
        num_examples = len(features)
        W1 = np.random.randn(self.input_dim, self.hidden_layer_dim) / np.sqrt(self.input_dim)
        b1 = np.zeros((1, self.hidden_layer_dim))
        W2 = np.random.randn(self.hidden_layer_dim, self.output_dim) / np.sqrt(self.hidden_layer_dim)
        b2 = np.zeros((1, self.output_dim))
        self.model={}
        for epoch in range(self.epochs):
             print("epoch : ", epoch+1)
             i = 0
             while i < num_examples:
                 features_batch = features[i:i+self.batch_size,::]
                 labels_batch = labels[i:i+self.batch_size]
                 i += self.batch_size
             # Forward propagation
                 z1 = features_batch.dot(W1) + b1
                 if self.activation_function == 1 :
                     a1 = np.tanh(z1)
                 elif self.activation_function == 2 :
                     a1 = self.sigmoid(z1)
                 else :
                     a1 = self.ReLu(z1)
                 z2 = a1.dot(W2) + b2
                 exp_scores = np.exp(z2)
                 
                 probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
                 delta3 = probs
                 delta3[range(len(features_batch)), labels_batch] -= 1
                 dW2 = (a1.T).dot(delta3)
                 db2 = np.sum(delta3, axis=0, keepdims=True)
                 if self.activation_function == 1 :
                     delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
                 elif self.activation_function == 2 :
                     delta2 = delta3.dot(W2.T) * np.multiply(a1, (1-a1))
                 else :
                     delta2 = delta3.dot(W2.T) * self.derivation_ReLu(a1)
                 delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
                 dW1 = np.dot(features_batch.T, delta2)
                 db1 = np.sum(delta2, axis=0)
             
             # Add regularization terms (b1 and b2 don't have regularization terms)
                 dW2 += self.reg_lambda* W2
                 dW1 += self.reg_lambda* W1
             
             # Gradient descent parameter update
                 W1 += -self.epsilon * dW1
                 b1 += -self.epsilon * db1
                 W2 += -self.epsilon * dW2
                 b2 += -self.epsilon * db2
             #annealing the epsilon
             if annealling == True:
                 self.epsilon = self.epsilon*np.exp( - epoch)
        
             # Assign new parameters to the model
             self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
             # Optionally print the loss.
             # This is expensive because it uses the whole dataset, so we don't want to do it too often.
             if print_info and epoch % 1000 == 0:
                print("Loss after iteration %i: %f" %(epoch, self.calculate_loss(features,labels,self.model)))
            
    def infer(self,feature):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # Forward propagation
        z1 = feature.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
