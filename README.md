# DSC550_Final
Final Term Project

## Dataset

categorized-comments.jsonl - size 500 MB

## Notes Related to Data Folder and Dataset Size

The dataset used here is categorized-comments.jsonl which is about 500 MB in size. Hence not able to upload in the data folder
as the git hub is throwing an error that the size is greater than 100 MB. However, the WorkflowQueue has a variable "datafilepath" for file path. If the variable is replaced with the right folder path where the data file is located on a given computer, the code should execute and generate the report

## Overview

The categorized-comments.jsonl data was used and analyzed using Logistic Regression and Naive bayes Classification
The goal of the analysis is to train the logistic and naive Bayes algorithms to accurately classify the text and show whether 
a given belongs to sports, video games, science and technology or news category.Finally the accuracy obtained by using both the methods was printed out in the form of markdown report in the Reports folder. Furthur, the trained models were pickled and stored in Models folder.

## Notes About The Project

The code is split between Workflows and WorkflowQueue. In the workflows all the independent tasks were defined. 
On the otherhand WorkflowQueue handles all the tasks together. This will furthur generate the markdown report that
includes information about accuracy
