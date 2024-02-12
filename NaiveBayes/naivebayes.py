# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:00:57 2022

@author: aymen
"""

import numpy as np 
#the naiveBayes model takes as input the different distribution that are needed to reprensent the input data(this work is being done in the preprocessing step)
class NaiveBayes():
    def __init__(self,classes,vocab,numdoc,min_count=1):
        self.min_count=min_count #minimal count of tokens
        self.vocab=vocab # the set of the input vocabulary
        self.classes = classes #the classes that have to be predicted after
        self.priors = {}
        self.conditionals = {}
        self.numdoc=numdoc #number of document or rows in the input training data 
        

    def learn(self):
        logspace_num_docs = np.log(self.numdoc)
        for classe in self.classes:
            # calculate priors: doc frequency
            self.priors[classe] = np.log(self.classes[classe]['doc_counts']) - logspace_num_docs

            # calculate conditionals
            self.conditionals[classe] = {}
            cdict = self.classes[classe]['terms']
            terms_in_class = sum(cdict.values())

            for term in self.vocab:
                t_ct = 1.  # We assume each term to occur in each document at least once - smoothing!
                t_ct += cdict[term] if term in cdict else 0.
                self.conditionals[classe][term] = np.log(t_ct) - np.log(terms_in_class + len(self.vocab))
    def infer(self, token_list: list):
        scores = {}
        for c in self.classes:
            scores[c] = self.priors[c]
            for term in token_list:
                if term in self.vocab:
                    scores[c] += self.conditionals[c][term]
        return max(scores, key=scores.get)
#this model variant is used in the case when the input data type is continous(floats)
# this model predicts the label class based on the normal distribution with mean and variance calculated based on the input data  
class NaiveBayescontinues:
    def __init__(self):
        pass
    def initialise(self,n_classes, n_features):
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)
        return self.mean,self.var,self.priors
    
    def learn(self, feature, target):
        n_samples= feature.shape[0]
        n_features= feature.shape[1]
        self.classes = np.unique(target)
        n_classes = len(self.classes)
        self.mean,self.var,self.prior=self.initialise(n_classes, n_features)
        
        for i, c in enumerate(self.classes):
            X_c = feature[target == c]
            self.mean[i, :] = X_c.mean(axis=0)
            self.var[i, :] = X_c.var(axis=0)
            self.priors[i] = X_c.shape[0] / float(n_samples)

    def predict(self, feature):
        pred = [self.calculate_posterior(x) for x in feature]
        return np.array(pred)

    def calculate_posterior(self, features):
        posteriors = []

        # calculate posterior probability for each class
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            posterior = np.sum(np.log(self.pdf(i, features)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def pdf(self, class_id, features):
        mean = self.mean[class_id]
        var = self.var[class_id]
        numerator = np.exp(-((features - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
