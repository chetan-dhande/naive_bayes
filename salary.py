# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:36:56 2020

@author: Chetan
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB

train = pd.read_csv("SalaryData_Train.csv")
test = pd.read_csv("SalaryData_Test.csv")

df = pd.concat([train,test],sort= False)


# Categorical boolean mask
categorical_feature_mask = df.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = df.columns[categorical_feature_mask].tolist()

df.info()

# instantiate labelencoder object
le = LabelEncoder()

# apply le on categorical feature columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
df[categorical_cols].head(10)

#Changing the target variable datatype from numerical to category
df['Salary'] = df['Salary'].astype('object')


#numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#newdf = df.select_dtypes(include=numerics)
#newdf.head()
#newdf.columns

for col in ['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']:
    df[col] = df[col].astype('category')

df.info()

train=df[:30161]
test=df[30161:]


X = train[['age', 'workclass', 'education', 'educationno', 'maritalstatus',
       'occupation', 'relationship', 'race', 'sex', 'capitalgain',
       'capitalloss', 'hoursperweek', 'native']]

y = train['Salary']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ignb = GaussianNB() 

pred_gnb = ignb.fit(X_train, y_train).predict(X_test)

print(confusion_matrix(y_test, pred_gnb))

from sklearn.metrics import classification_report
print(classification_report(y_test, pred_gnb))
ignb.score(X_train,y_train)
ignb.score(X_test,y_test)

X_new = test[['age', 'workclass', 'education', 'educationno', 'maritalstatus',
       'occupation', 'relationship', 'race', 'sex', 'capitalgain',
       'capitalloss', 'hoursperweek', 'native']]

y_new = test['Salary']

ignb.score(X_new,y_new)

ig = MultinomialNB()
ig.fit(X_train,y_train)


print(confusion_matrix(y_test, ig.predict(X_test)))

ig.score(X_train,y_train)
ig.score(X_test,y_test)
ig.score(X_new,y_new)
