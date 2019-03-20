#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:55:05 2019

@author: rohitmadan
"""
#Load data libray from skitlern
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd

d1= load_breast_cancer()
d=d1.data
target = d1.target

df=pd.DataFrame(data=d)
normalized_df=preprocessing.normalize(df)

print(df)
print(normalized_df)

df_y=pd.DataFrame(data=target)

X=normalized_df
y=df_y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(d1.feature_names)
feature_imp = pd.Series(clf.feature_importances_,index=d1.feature_names).sort_values(ascending=False)
print(feature_imp)





