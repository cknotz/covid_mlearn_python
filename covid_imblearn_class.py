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
#from sklearn.datasets import make_classification
import seaborn as sns


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

# Income
income = pd.get_dummies(df.res_income)
income.head()

df = df.join(income)
df.drop(['res_income'],
        axis = 1,
        inplace = True)

# Employment status
work = pd.get_dummies(df.r_work_before)
work.head()

df = df.join(work)
df.drop(['r_work_before'],
        axis = 1,
        inplace = True)

# Education
educ = pd.get_dummies(df.resp_edu)
educ.head()

df = df.join(educ)
df.drop(['resp_edu'],
        axis = 1,
        inplace = True)


# Gender
df['resp_gender'].head()
df.resp_gender = df.resp_gender.map({'Female':0,
                              'Male':1})

df['resp_gender'].head()
df = df.rename(columns = {'resp_gender':'male'})

# Region
df['resp_region'].head()
df.resp_region = df.resp_region.map({'french':1,
                                     'german':0})

df['resp_region'].head()
df = df.rename(columns = {'resp_region':'french'})


# Diagnosis
df['resp_diag'].head()
df.resp_diag = df.resp_diag.map({'Yes':1,
                                 'No':0})

list(df.columns)

# Split off labels
labels = df.resp_diag.copy()
labels.head()

features = df.drop(['resp_diag'],
                   axis = 1)
features.head()

# Test-train split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    test_size = .2,
                                                    random_state=17)

# Check result
print("Number of training samples:", len(X_train))
print("Number of test samples:", len(X_test))


# Analyze class imbalance
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

# Logistic dry run
