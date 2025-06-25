import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error 
import pandas as pd

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"
df = pd.read_csv(URL,header=None)
#selecting a single feature
#using 100 instances for simplicity
X = df.loc[:100,5].values
Y = df.loc[:100,13].values #target label
# print(X.shape)
# print(Y.shape)
X_reshaped = X[:,np.newaxis]
Y_reshaped = Y[:,np.newaxis]
# print(X_reshaped.shape)
# print(Y_reshaped.shape)

#intantiating the lassp regression model 
lasso = Lasso(alpha = 10)

#training the model
lasso.fit(X_reshaped,Y_reshaped)

#making predictions
Y_pred = lasso.predict(X_reshaped)
mse = mean_squared_error(Y_reshaped,Y_pred)
#evaluating the model
print(f"Mean Squared Error: {mse}")
print(f"Model Coefficients: {lasso.coef_}n")

#Plotting the results
plt.scatter(X_reshaped, Y_reshaped, color='blue', label='Actual')
plt.plot(X_reshaped, Y_pred, color='red', label='Predicted')
plt.xlabel('Feature (Column 5)')
plt.ylabel('Target (Column 13)')
plt.title('Lasso Regression')
plt.legend()
plt.show()
