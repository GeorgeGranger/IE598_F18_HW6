#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:30:59 2018

@author: huangsida
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np



iris = datasets.load_iris()
X = iris.data
y = iris.target

tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=5, 
                             random_state=1)
scores1=[]
scores2=[]
for a in range(1,11):
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=a, stratify=y)
    
    
    tree.fit(X_train, y_train)
    
    print('random_state= %.1f' %a)
    y_pred = tree.predict(X_train)
    scores1.append(accuracy_score(y_train, y_pred))
    print('Accuracy_train: %.2f' % accuracy_score(y_train, y_pred))
    y_pred = tree.predict(X_test)
    scores2.append(accuracy_score(y_test, y_pred))
    print('Accuracy_test: %.2f\n' % accuracy_score(y_test, y_pred))

print('In-sample mean and standard deviation: %.3f +/- %.3f' % (np.mean(scores1),
                                      np.std(scores1)))
print('Out-of-sample mean and standard deviation: %.3f +/- %.3f\n' % (np.mean(scores2),
                                      np.std(scores2)))
    

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=tree,
                         X=X,
                         y=y,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)

print('CV accuracy scores mean and standard deviation: %.3f +/- %.3f\n' % (np.mean(scores),
                                      np.std(scores)))
    
print("My name is Huang Sida")
print("My NetID is sidah2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
    

    
    
    

    

 
    

    
