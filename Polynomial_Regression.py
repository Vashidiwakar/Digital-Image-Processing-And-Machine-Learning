#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datasets
data_set = pd.read_csv("Salary_data.csv")
# print(data_set.head())

#Extracting dependent and independent variable
X_data = data_set.iloc[:,1:2].values
Y_data = data_set.iloc[:,2].values
# print(X_data)
# print(Y_data)

#fitting the linear regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regs = LinearRegression()
linear_regs.fit(X_data,Y_data)

#visulaizing the result for linear regression model
# plt.scatter(X_data,Y_data,color = "blue")
# plt.plot(X_data,linear_regs.predict(X_data),color = "red")
# plt.title("Bluff Detection Model(Linear Regression)")
# plt.xlabel("Position")
# plt.ylabel("Salary")
# plt.show()
#Fitting the polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
Polynomial_regs = PolynomialFeatures(degree=2)
X_poly = Polynomial_regs.fit_transform(X_data)
# print(X_poly)
linear_regs2 = LinearRegression()
linear_regs2.fit(X_poly,Y_data)

#visulaizing the result for linear regression model
plt.scatter(X_data,Y_data,color = "blue")
plt.plot(X_data,linear_regs2.predict(Polynomial_regs.fit_transform(X_data)),color = "red")
plt.title("Bluff Detection Model(Polynomial Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
