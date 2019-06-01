#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 06:53:46 2019

@author: ganga
"""
# The goal of Task1 is to load the jsonl file


# Importing the necessary libraries
import pandas as pd
import json
import pandas as pd
from pandas import DataFrame


# Method to read the jsonl file
def read_data(datafilepath):
    return pd.read_json(datafilepath, lines = True)




