# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 14:44:19 2022

@author: aymen
"""

import numpy as np
from helper_functions import _entropy,pp_float_list,_gini,_misclass,get_split_attribute
from helper_functions import most_common_class,impurity_reduction

np.set_printoptions(precision=3)

class DecisionNode(object):
    NODEID = 0

    def __init__(self, attr=-1, children=None, label=None):
        self.attr = attr
        self.children = children
        self.label = label
        self.id = DecisionNode.NODEID
        DecisionNode.NODEID += 1

        
class DecisionTreeID3(object):
    def __init__(self, criterion=_entropy):
        """
        :param criterion: The function to assess the quality of a split
        """
        self.criterion = criterion
        self.root = None
        
    def fit(self, X, y, verbose=0):

        def _fit(X, y, attributes=None):
            
            # Set up temporary variables
            N, d = X.shape
            if attributes == None:
                 attributes = {a_i: np.unique(X[:, a_i]) for a_i in range(d)}
            depth = d - len(attributes) + 1
            

            #if len(X) == 0: return DecisionNode()
            
            label, fIsPure = most_common_class(y)
            # Stop criterion 1: Node is pure -> create leaf node
            if fIsPure: 
                if verbose: print ("\t\t Leaf Node with label %s due to purity." % label)
                return DecisionNode(label=label)
            
            # Stop criterion 2: Exhausted attributes -> create leaf node
            if len(attributes) == 0:
                if verbose: print ("\t\t Leaf Node with label %s due to exhausted attributes." % label)
                return DecisionNode(label=label)
            
            # Get attribute with maximum impurity reduction
            a_i, a_ig = get_split_attribute(X, y, attributes, self.criterion, verbose=verbose)
            if verbose: print ("Level %d: Choosing attribute %d out of %s with gain %f" % (depth, a_i, attributes.keys(), a_ig))
            
            values = attributes.pop(a_i)
            splits = [X[:,a_i] == v for v in values]
            branches = {}
            
            for v, split in zip(values, splits):
                if not np.any(split):
                    if verbose: print ("Level %d: Empty split for value %s of attribute %d" % (depth, v, a_i))
                    branches[v] = DecisionNode(label=label)
                else: 
                    if verbose: print ("Level %d: Recursion for value %s of attribute %d" % (depth, v, a_i))
                    branches[v] = _fit(X[split,:], y[split], attributes=attributes)
            
            attributes[a_i] = values
            return DecisionNode(attr=a_i, children=branches, label=label)
   
        self.root = _fit(X, y)
        return self

    def predict(self, X):
        def _predict(x, node):
            if not node.children:
                return node.label
            else:
                v = x[node.attr]
                child_node = node.children[v]
                return _predict(x, child_node)

        return [_predict(x, self.root) for x in X]
    
    def print_tree(self, ai2an_map, ai2aiv2aivn_map):
        """
        ai2an_map: list of attribute names
        ai2aiv2aivn_map: list of lists of attribute values, 
                         i.e. a value, encoded as integer 2, of attribute with index 3 has name ai2aiv2aivn_map[3][2] 
        """
        
        def _print(node, test="", level=0):
            """
            node: node of the (sub)tree
            test: string specifying the test that yielded the node 'node'
            level: current level of the tree
            """

            prefix = ""
            prefix += " " * level
            prefix += " |--(%s):" % test
            if not node.children:
                print("%s assign label %s" % (prefix, ai2aiv2aivn_map[6][node.label]))
            else:
                print("%s test attribute %s" % (prefix, ai2an_map[node.attr]))
                for v, child_node in node.children.items():
                    an = ai2an_map[node.attr]
                    vn = ai2aiv2aivn_map[node.attr][v]
                    _print(child_node,"%s=%s" % (an, vn), level+1)
        
        return _print(self.root)
    
    def depth(self):
        def _depth(node):
            if not node.children:
                return 0
            else:
                return 1 + max([_depth(child_node) for child_node in node.children.values()])
        return _depth(self.root)