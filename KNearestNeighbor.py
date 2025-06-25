import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
#loading data
iris = datasets.load_iris()
#features
x = iris["data"]
y = iris["target"]
# print(iris["target"].ndim)
#Training the classifier
clf = KNeighborsClassifier()
clf.fit(x,y)
example = clf.predict([[1,1,1,1]])
print(example)

