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

df.head()

# Checking for missings
df.isna().any().any()

df.isna().sum().sum() # no
# 'income' contains non-answers, but not coded as missings!

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
df.columns

df.dtypes

# Gender
gender = pd.get_dummies(df['resp_gender'])
df = pd.concat([df,gender], axis = 1, sort = False)
del df['resp_gender']

# Region
region = pd.get_dummies(df['resp_region'])
df = pd.concat([df,region], axis = 1, sort = False)
del df['resp_region']

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

# Diagnosis
diag_mapper = {'No': 0,
               'Yes': 1}
df['diag'] = df['resp_diag'].replace(diag_mapper)
df['diag'].value_counts()
del df['resp_diag']

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

df.columns
df.dtypes # looks ok

# Convert to np array
features = df.drop('diag', axis = 1)
features

labels = df['diag']

features = np.array(features)
labels = np.array(labels)
labels
features

# Split into test, validation, and training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size = .2,
                                                    random_state= 17)

print("Number of training samples:", len(X_train))
print("Number of test samples:", len(X_test))

counts = np.bincount(y_test)
print(
    "Number of positive samples in test data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(y_test)
    )
)
counts = np.bincount(y_train)
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(y_train)
    )
)
# consider adjusting seed?


# Random undersampling - skipped because N of pos cases too small
#################################################################


# Random oversampling
#####################
