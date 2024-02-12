# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:56:56 2022

@author: aymen
"""

import numpy as np
def Accuracy(a,b):
    return sum(A==B for A,B in zip(a,b))/len(a)*100
def confusion_matrix(a,b):
    conf_matrix = np.eye(len(np.unique(a)))
    a_unique_labels = np.sort(np.unique(a))
    i = 0
    j = 0
    for y_pred in a_unique_labels:
        for y in a_unique_labels:
            conf_matrix[i,j] = sum(A == y and B == y_pred for A,B in zip(a,b))
            j += 1
        i += 1
        j = 0
    return conf_matrix , a_unique_labels
def precision(y,y_pred):
    conf_matrix,Labels = confusion_matrix(y, y_pred)
    precision = np.empty_like(Labels)
    i = 0 
    for Y in Labels:
        total_P= sum(A for A in conf_matrix[i,:])
        TP = conf_matrix[i,i]
        FP = total_P - TP
        precision[i] = TP/(TP+FP)*100
        i += 1
    return precision,Labels
def recall(y,y_pred):
    conf_matrix,Labels = confusion_matrix(y, y_pred)
    recall = np.empty_like(Labels)
    i = 0 
    for Y in Labels:
        total_P= sum(A for A in conf_matrix[:,i])
        TP = conf_matrix[i,i]
        FN = total_P - TP
        recall[i] = TP/(TP+FN)*100
        i += 1
    return recall,Labels
def Accuracy_conf(y,y_pred):
    conf,Labels=confusion_matrix(y, y_pred)
    Total=np.sum(conf,axis=(0,1))
    print(Total)
    true = 0
    for i in range(len(Labels)):
        true += conf[i,i]
    return np.round(true/Total*100,2)
def F1_score(y,y_pred):
    preci,_ = precision(y,y_pred)
    recal, _ = recall(y, y_pred)
    return np.round((2*preci*recal)/(preci+recal),2)
    
               
def rss_value(actuals, forecasted):

    residuals = actuals - forecasted
    ## Squared each residual
    squared_residuals = [np.power(residual, 2) for residual in residuals]
    rss = sum(squared_residuals)
    return rss


## Total sum of square
def tss_value(actuals):

    ## Calcuate mean
    actual_mean = actuals.mean()
    ## Squared mean difference value
    mean_difference_squared = [np.power(
    (actual - actual_mean), 2) for actual in actuals]
    tss = sum(mean_difference_squared)
    return tss


## R-squared value
def r_squared_value(actuals, forecasted):

    rss = rss_value(actuals, forecasted)
    tss = tss_value(actuals)
    ## Calculating R-squared value
    r_squared_value = 1 - (rss/float(tss))
    return np.around(r_squared_value, 2)