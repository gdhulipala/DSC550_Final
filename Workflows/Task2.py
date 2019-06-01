#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 07:10:33 2019

@author: ganga
"""
# The goal of Task2 is to pre process the data
# Task2 calls task1 to get the loaded data
# Once the loaded data is accessed, the string values in th etarget variable is converted into categories
# Furthur the data is condensed to 80,000 rows
# The data is selected in such a way that the target categorical variables have equal representation
# Once the data is condensed, this task asl handles the 

# Importing the tasks from the workflows folder
from Workflows import Task1
from Workflows.Task1 import read_data
# Import necessary libraries
import re
import pandas as pd
import json

# Method to load the data
# This method also preprocess the data interms of handling categorical variables and condensing data

def load_processdata(datafilepath):
    
    data1 =  Task1.read_data(datafilepath) 
    # Encode the categorical variable of categorical variable into integers
    data1.loc[data1["cat"]=="sports", "cat"]=0
    data1.loc[data1["cat"]=="science_and_technology", "cat"]=1
    data1.loc[data1["cat"]=="video_games", "cat"]=2
    data1.loc[data1["cat"]=="news", "cat"]=3
    # looking at the top five rows of the dataset
    data1.cat.astype(int).head()
    # Extracting 20,000 rows of each category from the total dataset
    data_0 = data1[data1["cat"]==0].iloc[0:20000]
    data_1 = data1[data1["cat"]==1].iloc[0:20000]
    data_2 = data1[data1["cat"]==2].iloc[0:20000]
    data_3 = data1[data1["cat"]==3].iloc[0:20000]
    
    # Merging the individual data frames from 2 million rows to 80,000 rows as the classifier has hard time fitting the data
    data_condensed = pd.concat([data_0, data_1], axis=0)
    data_condensed = pd.concat([data_condensed, data_2], axis=0)
    data_condensed = pd.concat([data_condensed, data_3], axis=0)
    return data_condensed

# Method to preprocess data interms of removing all non alphabateical characters
def commas(txt):
    txt = re.sub('[^a-zA-Z]', " ", txt)
    return txt

# Method to invoke both the preprocess methods mentioned above
def process(datafilepath):
    y = load_processdata(datafilepath) 
    y["txt"] = y["txt"].apply(commas)
    return y







    
      