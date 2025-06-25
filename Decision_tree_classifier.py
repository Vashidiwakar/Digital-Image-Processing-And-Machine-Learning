import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree

#loading the data
iris = load_iris()

#store the feature matrix (X) and the response vector (Y)
X = iris.data
Y = iris.target

#splitting X and Y into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

#training the model on training set
#import Perceptron from perceptron.py
DT = tree.DecisionTreeClassifier()
DT = DT.fit(X_train,Y_train)

#plot the decision tree
fn = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn = ['setosa','versicolor','virginica']
_,ax = plt.subplots(figsize = (15,15)) #Resize figure
tree.plot_tree(DT,feature_names = fn, class_names=cn,filled=True,ax=ax)