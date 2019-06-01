#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 03:45:34 2019

@author: ganga
"""

# The goal of Task4 is to get the vectorized data from task3 and fit the data to naivebayes regression

# Importing the necessary tasks
from Workflows import Task2
from Workflows import Task1
from Workflows import Task3
from Workflows.Task3 import vectorizer
# Importing the necessary libraries
import re
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Defining a method to get the data from Task3 
# Furthur the method also handles splitting the data into test and train split
# Split data is fit into logistic regression
# Calculates the accuracy score

def modelfit_naivebayes(datafilepath):
    # Getting the vectorized data
    X, y = Task3.vectorizer(datafilepath)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 1)
    # Splitting the data
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train.astype("int"))
    # Getting predictions by passing the test dataset
    test_predictions = classifier.predict(X_test)
    train_predictions = classifier.predict(X_train)
    # Calculating the accuracy by passing the test dataset
    accuarcy_test = accuracy_score(y_test.astype("int"), test_predictions.astype("int"))
    accuarcy_train = accuracy_score(y_train.astype("int"), train_predictions)
    # Defining the path to store the pickled model
    path_folder=path_folder= "/Users/ganga/Desktop/Data Science Masters/Courses/Data Mining/Week 8/DataSciencePipeline/Models/Naive.pkl"  
    # Pickling the model
    with open(path_folder, 'wb') as f:
         pickle.dump(classifier, f)
    return accuarcy_test



