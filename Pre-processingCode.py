#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Data.csv")
ds = dataset.iloc[:,:].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean') #try help
imputer.fit(ds[:,1:3]) #calculates mean
ds[:,1:3] = imputer.transform(ds[:,1:3]) #replace the nan with mean of the column
# print(ds)

#Encoding categorical data; Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
lbl_encoder = LabelEncoder()
ds[:,3] = lbl_encoder.fit_transform(ds[:,3])
# print(ds)

#Splitting the dataset into the Training set and Test set
#X = dataset.iloc[:,:-1].values
#Y = dataset.iloc[:,-1].values
X = ds[:,:-1]
Y = ds[:,-1]
# print(X)
# print(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1)
print(X_train)
print(X_test)
# print(Y_train)
# print(Y_test)

#Feature scaling
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
X_train[:,2:] = min_max_scaler.fit_transform(X_train[:,2:])

Standardisation = preprocessing.StandardScaler()
X_test[:,2:] = Standardisation.fit_transform(X_test[:,2:])

print(X_train)
print(X_test)