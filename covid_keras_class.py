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

# convert four-digit postcode to one-digit, then dummy-code
df['ZIP'].describe()

df['zip1'] = df["ZIP"].astype(str).str[:1].astype(int)
df['zip1'].unique()
del df['ZIP']

zip = pd.get_dummies(df['zip1'])
zip
df = pd.concat([df,zip], axis = 1, sort = False)
del df['zip1']

# Check categorical vars (gender, diagnosis, region, r_work before)
df.dtypes

# Gender
df['male'] = pd.get_dummies(df['resp_gender'])
df['male'].dtypes
del df['resp_gender']

# Region
df['german'] = pd.get_dummies(df['resp_region'])
df['german'].dtypes
del df['resp_region']

# Diagnosis
df['notdiag'] = pd.get_dummies(df['resp_diag'])
df['notdiag'].dtypes
del df['resp_diag']

# Employment status
work = pd.get_dummies(df['r_work_before'])
work
df = pd.concat([df,work], axis = 1, sort = False)
del df['r_work_before']

# Income
inc = pd.get_dummies(df['res_income'])
inc
df = pd.concat([df,inc], axis = 1, sort = False)
del df['res_income']
# Note: Income is treated as nominal rather than ordinal to avoid losing around 300 obs that refused to state their income;

# Ordinal variable, education
df['resp_edu'].value_counts()
edu_mapper = {'no degree': 0,
              'primary': 1,
              'secundary': 2,
              'vocational training': 3,
              'upper sec./upper voc.': 4,
              'univ. appl.': 5,
              'university': 6}

df['edu'] = df['resp_edu'].replace(edu_mapper)
df['edu'].unique()
del df['resp_edu']

df.dtypes # looks ok

# Preprocessing
###############
import sklearn as sk

# Standardize

# Convert to np array


##
