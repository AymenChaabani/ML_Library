# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 10:46:06 2022

@author: aymen
"""
import numpy as np
def pp_float_list(ps):#pretty print functionality
    return ["%2.3f" % p for p in ps]
def _gini(p):
    """
    p: class frequencies as numpy array with np.sum(p)=1
    returns: impurity according to gini criterion
    """
    return 1. - np.sum(p**2)

def _entropy(p):
    """
    p: class frequencies as numpy array with np.sum(p)=1
    returns: impurity according to entropy criterion
    """
    idx= np.where(p==0.) #consider 0*log(0) as 0
    p[idx] = 1.
    r=p*np.log2(p)
    return -np.sum(r)
    
def _misclass(p):
    """
    p: class frequencies as numpy array with np.sum(p)=1
    returns: impurity according to misclassification rate
    """
    return 1-np.max(p)
def get_split_attribute(X, y, attributes, impurity, verbose=0):
    """
    X: data matrix n rows, d columns
    y: vector with n rows, 1 column containing the target concept
    attributes: A dictionary mapping an attribute's index to the attribute's domain
    impurity: impurity function of the form impurity(p_1....p_k) with k=|y.unique|
    returns: (1) idx of attribute with maximum impurity reduction and (2) impurity reduction
    """
    
    N, d = X.shape

    IR = [0.] * d
    for a_i in attributes.keys():
        IR[a_i] = impurity_reduction(X, a_i, y, impurity, verbose)
    if verbose: print("Impurity reduction for class attribute (ordered by attributes)",(pp_float_list(IR)))
    b_a_i = np.argmax(IR)
    return b_a_i, IR[b_a_i]
def most_common_class(y):
    """
    :param y: the vector of class labels, i.e. the target
    returns: (1) the most frequent class label in 'y' and (2) a boolean flag indicating whether y is pure
    """
    y_v, y_c = np.unique(y, return_counts=True)
    label = y_v[np.argmax(y_c)]
    fIsPure = len(y_v) == 1
    return label, fIsPure
def impurity_reduction(X, a_i, y, impurity, verbose=0):
    """
    X: data matrix n rows, d columns
    a_i: column index of the attribute to evaluate the impurity reduction for
    y: concept vector with n rows and 1 column
    impurity: impurity function of the form impurity(p_1....p_k) with k=|X[a].unique|
    returns: impurity reduction
    Note: for more readable code we do not check any assertion 
    """
    N_rows=float(X.shape[0])
    N_features=float(X.shape[1])
    
    y_v = np.unique(y)
    
    # Compute relative frequency of each class in X
    p = (1. / N_rows) * np.array([np.sum(y==c) for c in y_v])
    # ..and corresponding impurity l(D)
    H_p = impurity(p)
    
    if verbose: print ("\t Impurity %0.3f: %s" % (H_p, pp_float_list(p)))
    
    a_v = np.unique(X[:, a_i])
    
    # Create and evaluate splitting of X induced by attribute a_i
    # We assume nominal features and perform m-ary splitting
    H_pa = []
    for a_vv in a_v:
        mask_a = X[:, a_i] == a_vv
        N_a = float(mask_a.sum())
                     
        # Compute relative frequency of each class in X[mask_a]
        pa = (1. / N_a) * np.array([np.sum(y[mask_a] == c) for c in y_v])
        H_pa.append((N_a / N_rows) * impurity(pa))
        if verbose: print ("\t\t Impurity %0.3f for attribute %d with value %s: " % (H_pa[-1], a_i, a_vv), pp_float_list(pa))
    
    IR = H_p - np.sum(H_pa)
    if verbose:  print ("\t Estimated reduction %0.3f" % IR)
    return IR