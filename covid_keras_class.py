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
df.head()

# Checking for missings
df.isna().any().any()
df.isna().sum().sum() # no
# 'income' contains non-answers, but not coded as missings!

# Data mangling
###############

# convert four-digit postcode to one-digit
df['ZIP'].describe()

df['zip1'] = df["ZIP"].astype(str).str[:1].astype(int)
df['zip1'].unique()
del df['ZIP']

# Check categorical vars (gender, diagnosis, region, edu, r_work before, income)

df['resp_gender'].describe()
df['resp_diag'].describe()
df['res_income'].unique()

# Preprocessing
import sklearn as sk

##
