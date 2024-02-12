# -*- coding: utf-8 -*-
"""
Created on Sat May 28 12:53:11 2022

@author: aymen
"""
import os
import numpy as np 
import codecs
import re
def tokenize_str(doc):
    return re.findall(r'\b\w\w+\b', doc)# return all words with #characters > 1

def tokenize_file(doc_file):
    with codecs.open(doc_file, encoding='latin1') as doc:
        print(doc)
        doc = doc.read().lower()
        _header, _blankline, body = doc.partition('\n\n')
        return tokenize_str(body) # return all words with #characters > 1
def _build_vocab(path_base,min_count):
    classes={}
    vocabulary={}
    num_docs = 0
    
    for class_name in os.listdir(path_base):
        
        classes[class_name] = {"doc_counts": 0, "term_counts": 0, "terms": {}}
        path_class = os.path.join(path_base, class_name)
        for doc_name in os.listdir(path_class):
            terms = tokenize_file(os.path.join(path_class, doc_name))
            num_docs += 1
            classes[class_name]["doc_counts"] += 1

            # build vocabulary and count terms
            for term in terms:
                classes[class_name]["term_counts"] += 1
                if not term in vocabulary:
                    vocabulary[term] = 1
                    classes[class_name]["terms"][term] = 1
                else:
                    vocabulary[term] += 1
                    if not term in classes[class_name]["terms"]:
                        classes[class_name]["terms"][term] = 1
                    else:
                        classes[class_name]["terms"][term] += 1
    vocabulary = {k:v for k,v in vocabulary.items() if v > min_count}
    return classes,vocabulary,num_docs

    