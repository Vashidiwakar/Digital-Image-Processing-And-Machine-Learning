import matplotlib.pyplot as plt
import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000])

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_mdl = PolynomialFeatures(degree=2)
X_new = poly_mdl.fit_transform(X)
linear_rgs = LinearRegression()
linear_rgs.fit(X_new,Y)

plt.scatter(X,Y,color = 'blue',label = 'Data Points')
plt.plot(X,linear_rgs.predict(X_new),color ='red',label = 'Regression line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot 1')
plt.show()