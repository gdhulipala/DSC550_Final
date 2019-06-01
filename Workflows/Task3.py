#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 07:28:42 2019

@author: ganga
"""
# The goal of Task3 is to vectorize the dataset
# Importing the necessary tasks
from Workflows import Task2
from Workflows import Task1
from Workflows.Task2 import process

# Importing the necessary libraries
import re
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer


# Method to vectorize the data
# For simplication in execution the maximum features were kept to 1000
def vectorizer(datafilepath):
    d = Task2.process(datafilepath)  
    cv = CountVectorizer(stop_words = {'english'}, max_features = 1000)
    X = cv.fit_transform(d["txt"]).toarray()
    y = d.iloc[:, 0].values 
    return X,y    

    
