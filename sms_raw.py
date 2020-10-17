# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 19:39:29 2020

@author: Chetan
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB
df = pd.read_csv("D:\\chetan\\assignment\\15.naive bayes\\sms_raw_NB.csv",encoding = "ISO-8859-1")


import re
stop_words = []
with open("D:\\chetan\\assignment\\14text mining\\stop.txt") as f:
    stop_words = f.read()

stop_words = stop_words.split("\n")

def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []

    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

df.text = df.text.apply(cleaning_text)
def split_into_words(i):
    return [word for word in i.split(" ")]



from sklearn.model_selection import train_test_split

train,test = train_test_split(df,test_size=0.3)

df_bow = CountVectorizer(analyzer=split_into_words).fit(df.text)

all_df_matrix = df_bow.transform(df.text)
all_df_matrix.shape 

train_df_matrix = df_bow.transform(train.text)
train_df_matrix.shape



test_df_matrix = df_bow.transform(test.text)
test_df_matrix.shape 

classifier_mb = MB()
classifier_mb.fit(train_df_matrix,train.type)
train_pred_m = classifier_mb.predict(train_df_matrix)
np.mean(train_pred_m==train.type)

test_pred_m = classifier_mb.predict(test_df_matrix)
np.mean(test_pred_m==test.type)


classifier_gb = GB()
classifier_gb.fit(train_df_matrix.toarray(),train.type.values)
train_pred_g = classifier_gb.predict(train_df_matrix.toarray())
np.mean(train_pred_g==train.type) 

test_pred_g = classifier_gb.predict(test_df_matrix.toarray())
np.mean(test_pred_g==test.type) 


tfidf_transformer = TfidfTransformer().fit(all_df_matrix)

train_tfidf = tfidf_transformer.transform(train_df_matrix)


test_tfidf = tfidf_transformer.transform(test_df_matrix)
 
classifier_mb = MB()
classifier_mb.fit(train_tfidf,train.type)
train_pred_m = classifier_mb.predict(train_tfidf)
np.mean(train_pred_m==train.type)

test_pred_m = classifier_mb.predict(test_tfidf)
np.mean(test_pred_m==test.type) 

classifier_gb = GB()
classifier_gb.fit(train_tfidf.toarray(),train.type.values) 
train_pred_g = classifier_gb.predict(train_tfidf.toarray())
np.mean(train_pred_g==train.type) 
test_pred_g = classifier_gb.predict(test_tfidf.toarray())
np.mean(test_pred_g==test.type) 






