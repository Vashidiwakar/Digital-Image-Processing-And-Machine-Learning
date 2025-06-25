from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

#load the iris data
iris = datasets.load_iris()

#store the feature matrix (X) and response vector (Y)
# print(list(iris.keys()))
X = iris.data
Y = iris.target

#splitting X and Y into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 1)

svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel = 'poly', degree = 8)
#svclassifier = SVC(kernel = 'rbf')
#svclassifier = svc(kernel = 'sigmoid')
svclassifier.fit(X_train,Y_train)

#making predictions on the testing set
Y_pred = svclassifier.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# scores = cross_val_score(X,Y,cv = 5, scoring = 'accuracy') #k=5 fold

# print(scores)
# print("Mean ACC = ",scores.mean())