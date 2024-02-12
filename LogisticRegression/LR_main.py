# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:30:02 2022

@author: aymen
"""
import numpy as np
from logistic_regression import *
n_samples = 1000
mean = [5, 9]
cov = [[2.5, 0.8], [0.8, 0.5]]
X_p = np.random.multivariate_normal(mean, cov, n_samples).T
mean = [11, 3]
cov = [[3, -1.3], [-1.3, 1.2]]
X_n_1 = np.random.multivariate_normal([11, 3], cov, int(n_samples/2)).T
X_n_2 = np.random.multivariate_normal([5, 2], cov, n_samples-int(n_samples/2)).T
X_n = np.hstack([X_n_1, X_n_2])

XY_p = np.vstack([X_p, np.ones_like(X_p[0])])
XY_n = np.vstack([X_n, np.zeros_like(X_n[0])])
XY_n.shape


XY = np.hstack([XY_n, XY_p])
data_XY = np.copy(XY).T
np.random.shuffle(data_XY)
data_train = data_XY[:1600]

data_test = data_XY[:400]
data_train.shape

model=LogisticRegression(10,2,100)

feature=data_train[:,:2]
print("shapef",feature.shape)
target=data_train[:,-1]
print("shapet",target.shape)

tt=data_test[:,-1]
model.learn(feature, target)
y_pred=model.infer(data_test[:,:2])
j=0
k=0
for i in range(len(y_pred)):
    if (y_pred[i]==tt[i]):
        j+=1
    if(y_pred[i]==1):
        k+=1
print(j,k,len(y_pred))
#print("finally y: {} \n y pred {}".format(target,y_pred))


