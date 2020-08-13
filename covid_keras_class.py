# Classification of COVID-diagnoses using neural net for credit card default
############################################################################

# Carlo Knotz

# Importing packages
#!/usr/bin/env python3
import csv
import sys
import os
import numpy as np
import pandas as pd


# File path
print(os.getcwd())
parpath = os.path.dirname(os.path.abspath('.'))
path = os.path.join(parpath,
                    'R Analysis',
                    'covid_dashboard',
                    'dashdata.dta')
path

# Reading dataset
data = pd.read_stata(path)
data.head()

# Selecting relevant predictors
