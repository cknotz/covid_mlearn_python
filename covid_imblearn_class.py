##################################################
# Classification of COVID-diagnoses using imblearn
##################################################

# Carlo Knotz

# Based on: https://towardsdatascience.com/machine-learning-and-class-imbalances-eacb296e776f

# Importing packages
#!/usr/bin/env python3
import csv
import sys
import os
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib as plt


# Test case
#X,y = make_classification(n_samples=10000, n_features=2, n_informative=2,
#                          n_redundant=0, n_repeated=0, n_classes=2,
#                          flip_y=0, weights=[0.99,0.01], random_state=17)
# –> arrays!

# Data pre-processing from Keras approach
#########################################

# File path
print(os.getcwd())
parpath = os.path.dirname(os.path.abspath('.'))
path = os.path.join(parpath,
                    'R Analysis',
                    'covid_dashboard',
                    'dashdata.dta')
print(path)

# Reading dataset
data = pd.read_stata(path)
data.head()

# Selecting relevant predictors
data.columns

df = data[['resp_gender',
           'resp_age',
           'resp_edu',
           'resp_region',
           'resp_diag',
           'r_work_before',
           'ZIP',
           'res_income']]

del parpath, path, data # removing clutter

df.describe()
list(df.columns)
df.head()

# Checking for missings
df.isna().any().any()

df.isna().sum().sum() # no
# 'income' contains non-answers, but not coded as missings!

# Dummy coding categorical predictors
#####################################

# convert four-digit postcode to one-digit, then dummy-code
df['ZIP'].describe()

df['zip_num'] = df["ZIP"].astype(str).str[:1].astype(int)
df['zip_num'].unique()
del df['ZIP']
df['zip_num'].head()

zip = pd.get_dummies(df.zip_num,
                     prefix = 'zip')
zip.head()

df = df.join(zip)
df.drop(['zip_num'],
        axis=1,
        inplace = True)
