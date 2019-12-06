import pandas as pd
import numpy as np

#read the csv file and store it in the bank data frame
bank=pd.read_csv('bank-additional.csv',sep=';')
bank.head(10)

#list of columns for reference
bank.columns

#  y (response)
# convert the response to numeric values and store as a new column
bank['outcome'] = bank.y.map({'no':0, 'yes':1})

#import matplotlib
import matplotlib.pyplot as plt

## 1. For age
# probably not a great feature since lot of outliers
bank.boxplot(column='age', by='outcome')

## 2. For job
## useful features as all values revolve around same space
bank.groupby('job').outcome.mean()

# create job_dummies (we will add it to the bank DataFrame later)
job_dummies = pd.get_dummies(bank.job, prefix='job')
job_dummies.drop(job_dummies.columns[0], axis=1, inplace=True)

## 3.default
# looks like a useful feature
bank.groupby('default').outcome.mean()

# so, let's treat this as a 2-class feature rather than a 3-class feature
bank['default'] = bank.default.map({'no':0, 'unknown':1, 'yes':1})

## 4. contact
# convert the feature to numeric values
bank['contact'] = bank.contact.map({'cellular':0, 'telephone':1})

## 5. month
# looks like a useful feature at first glance
bank.groupby('month').outcome.mean()

# but, it looks like their success rate is actually just correlated with number of calls
# thus, the month feature is unlikely to generalize
bank.groupby('month').outcome.agg(['count', 'mean']).sort_values('count')

## 6. duration
# looks like an excellent feature, but you can't know the duration of a call beforehand, thus it can't be used in your model
bank.boxplot(column='duration', by='outcome')

## 7.1 previous
# looks like a useful feature
bank.groupby('previous').outcome.mean()

## 7.2 poucome
# looks like a useful feature
bank.groupby('poutcome').outcome.mean()

# create poutcome_dummies
poutcome_dummies = pd.get_dummies(bank.poutcome, prefix='poutcome')
poutcome_dummies.drop(poutcome_dummies.columns[0], axis=1, inplace=True)
# concatenate bank DataFrame with job_dummies and poutcome_dummies
bank = pd.concat([bank, job_dummies, poutcome_dummies], axis=1)

# prepare a boxplot on euribor3m by outcome, and comment on the 'euribor3m' feature
# looks like an excellent feature
bank.boxplot(column='euribor3m', by='outcome')

feature_cols = ['default', 'contact', 'previous', 'euribor3m'] + list(bank.columns[-13:])
X = bank[feature_cols]
# create y
y = bank.outcome
X.head()
