#Classifier which classifies from the given feature whether it is a virginica(1) or not(0)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
#loading data 
iris = datasets.load_iris()
# print(list(iris.keys()))
#feature
x = iris["data"][:,3:]
#label
y = (iris["target"]==2).astype(int)
#Training the logistic regressor
clf = LogisticRegression()
clf.fit(x,y)
example = clf.predict([[1.6]])
print(example)
#using matplotlib for visualisations
x_new = np.linspace(0,3,1000).reshape(-1,1)
# print(x_new)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1],"g-",label = "virginica")
plt.legend()
plt.xlabel("X")
plt.ylabel("Probability")
plt.show()
