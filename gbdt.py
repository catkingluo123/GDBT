# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:42:15 2017

@author: Administrator
"""
import numpy as np
import csv
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
def loadCsv(filename):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return np.array(dataset)  #list
def main():
    gbdt=GradientBoostingClassifier(
            init=None,
            learning_rate=0.1,
            loss='deviance',
            max_depth=7,
            max_features=None,
            max_leaf_nodes=None,
            min_samples_leaf=1,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=100,
            random_state=None,
            subsample=1.0,
            verbose=0,
            warm_start=False)
    filename = 'pima-indians-diabetes.data.csv'
    dataset = loadCsv(filename)
   # X = dataset[:][0:8]
    X = dataset[:,0:8]
    y = dataset[:,8:9]
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_test_minmax = min_max_scaler.transform(X_test)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    gbdt.fit(X_train_minmax,y_train)
    print(gbdt.score(X_test_minmax,y_test))
    #print(gbdt.predict(X_test))
main()
