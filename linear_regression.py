#importing the required modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
#loading data from sklearn datasets
diabetes = datasets.load_diabetes()
#feature
# print(diabetes['data'].shape)
diabetes_X = diabetes.data[:,2].reshape(-1,1)
# print(diabetes_X.shape)
#diabetes_X = diabetes.data[:,np.newaxis,2]
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]
#[:-30]: This slice includes all elements from the beginning of the array up to, but not including, the 30th element from the end.
#[-30:]: This slice includes all elements from the 30th element from the end to the last element of the array
#label
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]
#Training the linear regression model
model = linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_predicted = model.predict(diabetes_X_test)
print("Mean squared error is:",mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))
print("Weights:",model.coef_)
print("intercept:",model.intercept_)
# using matplotlib for visualisation
plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)
plt.show()