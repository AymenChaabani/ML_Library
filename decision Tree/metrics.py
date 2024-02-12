# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 10:57:32 2022

@author: aymen
"""
import numpy as np
def acc(y, y_p):
    correct = y == y_p
    acc = np.sum(correct) / float(len(y))
    return acc

def err_mis(y, y_p):
    return 1. - acc(y, y_p)