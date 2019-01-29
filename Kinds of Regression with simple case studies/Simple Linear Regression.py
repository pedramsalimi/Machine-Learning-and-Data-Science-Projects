#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 08:32:44 2018

@author: Pedram
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
#Splitting dataset to train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
#Fitting Simple Linear Regression to the training set which handle **Feature Scaling** itself
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#Predicting the test set results
y_pred = regressor.predict(X_test)
#Visualizing training set results
plt.scatter(X_train, y_train, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
#Visualizing test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
