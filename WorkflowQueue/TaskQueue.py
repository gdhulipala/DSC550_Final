#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 04:03:06 2019

@author: ganga
"""

#--------------- TasksQueue.py HANDLES THE COMPLETE WORKFLOW BY HANDLING ALL THE INDIVIDUAL TASKS SPECIFIED IN THE WORKFLOWS FOLDER------------------------------------------------------

# By running this file, it generates pickle files and the combined report for the two models tested
# The strategy is to fit the categorized-comments-json file into two classification models i.e. Logistic and Naive Bayes
# The main function triggers the method workflowqueue by passing the path of the data file
# Once the workflowqueue method is invokes it will call all the necessary tasks from the workflows folder
# Markdown report will be created in the reports folder
# Models folder gets all the pickled algorithms
# Data folder has the data file

#------WORKFLOWS FOLDER--------------------------------------------------------

# Task1 - Loads the Data
# Task2 - PreProcess the data
# Task3 - Vectorization
# Task4 - Fitting to Logistic Regression and pickling
# Task5 - Fitting to Logistic Naive Bayes and pickling

# --------------WORKFLOW QUEUE FOLDER------------------------------------------

# TasksQueue - Queues all the tasks and genearates a markdown file

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Importing all the necessary tasks from the workflows folder

from Workflows import Task2
from Workflows import Task1
from Workflows import Task3
from Workflows import Task4
from Workflows import Task5

# Importing all the necessary libraries
from Workflows.Task4 import modefit_logistic
from Workflows.Task5 import modelfit_naivebayes
import re
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Specifying the location of the data
datafilepath = "/Users/ganga/Desktop/Data Science Masters/Courses/Data Mining/Week 8/DataSciencePipeline/Data/categorized-comments.jsonl"

# Defining the method that handles all the tasks and generates all the necessary pickle files and markdown report
def workflowqueue(datafilepath):
    # Calling both the algorithms from the corresponding task files
    Accuracy_Logistic = Task4.modefit_logistic(datafilepath)
    Accuracy_NaiveBayes = Task5.modelfit_naivebayes(datafilepath)
    # Below is the path on where to put the markdown file once created
    path_folder= "/Users/ganga/Desktop/Data Science Masters/Courses/Data Mining/Week 8/DataSciencePipeline/Report/Accuracy Report.md" 
    # Specifying the data to be included in the markdown file
    with open(path_folder, 'w') as f:
         f.write('## {}'.format("Accuracy Values For The Test Sets"))
         f.write('\n\n# {}{} {}'.format("Accuracy Logistic Regression",":", Accuracy_Logistic))
         f.write('\n\n# {}'.format('Parameters Used'))
         f.write('\n* {} {}'.format('Penalty: ', "l2"))
         f.write('\n* {} {}'.format('Solver: ', "sag"))
         f.write('\n* {} {}'.format('MultiClass: ', "Multinomial"))
         f.write('\n\n\n\n# {}{} {}'.format("Accuracy Naive Bayes", ":", Accuracy_NaiveBayes))
         f.write('\n\n# {}'.format('Parameters Used'))
         f.write('\n* {} {}'.format('Priors: ', "None"))
    return Accuracy_Logistic, Accuracy_NaiveBayes


# Main function invokes the workflowqueue method
def main():
   workflowqueue(datafilepath)
        

# This will call the main function once this file is run
if __name__ == "__main__":
   main()  
