import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import *
import matplotlib.pyplot as plt
import patsy


# Load data
#url = 'http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv'
#data available from above web address


advertising = pd.read_csv('Advertising.csv')
del(advertising['Unnamed: 0'])

y, X = patsy.dmatrices('Sales ~ TV + Radio', data = advertising, return_type='dataframe') #use this matrix builder to apply transformations and interaction terms


Xwithones = sm.add_constant(X) #In case you want to add the arrays manually to the OLS model


model = sm.OLS(y, Xwithones)
fitted = model.fit()
print fitted.summary()

rse = fitted.mse_resid*.5 #Residual Standard Error
print 'RSE is %.4f' %rse

def model(feature, test, labelforx, labelfory, graphname): #only works for simple regression i.e. one feature, plots feature against fitted values
	plt.plot(feature, test, 'ro')
	plt.plot(feature, fitted.fittedvalues, 'b')
	plt.legend(['Data', 'Model Prediction'])
	#plt.ylim(
	#plt.xlim
	plt.xlabel(labelforx)
	plt.ylabel(labelfory)
	plt.title(graphname)

	plt.show()

def residuals(feature, test): # plots residuals and residuals histogram
	plt.plot(test - fitted.fittedvalues, 'bo')
	plt.plot(fitted.fittedvalues - fitted.fittedvalues, 'r')
	plt.legend(['Residuals'])
	plt.xlabel('Observation')
	plt.ylabel('Magnitude')
	plt.title('Plot of Residuals')
	plt.show()

def lev_and_influence():	# plots leverage and outliers

	plot_leverage_resid2(fitted) #leverage vs normalized residuals
	influence_plot(fitted) # studentized residuals vs leverage
	plt.show()

