#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 08:27:08 2018

@author: Pedram
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values #We iclude 2 just because of we want X as a matrix of features,
#by the way don't worry about it because the upper band is excluded and not inlcude the 2 ;-)
y = dataset.iloc[:,2].values

#There is no need to split the data set to train and test because we want a very accurate
#result and we have not muck info so we use all of the data as train set

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures #poly_reg is a transformer tool that
#transform our matrix of features X in to new matrix of features that we are gonna call
#X_poly which will be a new matrix of features containing not only the independent variables x
#but also x  power 2 or 3 ...
poly_reg = PolynomialFeatures(degree=3) #We change degree from 2 to 3 and the result became much better
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualizing the linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or bluff ( Linear Regression )')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#Visualizing the polynomial regression results

plt.scatter(X, y, color='red')
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
#plt.plot(X_poly, lin_reg_2.predict(X_poly), color='blue')
#don't use X_poly, if we want new observations here and plot some new polynomial regression
#results we rather poly_reg.fit_transform(X)
# for smoothing curve use X_grid instead X
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or bluff ( Polynomial Regression )')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predicting .. compare..
lin_reg.predict(6.5)
lin_reg_2.predict(poly_reg.fit_transform(6.5))





