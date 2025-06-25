import numpy as np
import matplotlib.pyplot as plt

#generate x and y
X = np.linspace(0,1,101)
Y = 1 + X + X*np.random.random(len(X))
# print(X)
# print(Y)

X1 = np.vstack([X, np.ones(len(X))]).T #combines two arrays
# print(X1)
# print(X.shape)
# print(X1.shape)

alpha = np.linalg.lstsq(X1,Y,rcond = None)[0] #calculates m and c
# np.linalg.inv(): Computes the inverse of a matrix.
# np.linalg.det(): Computes the determinant of a matrix.
# np.linalg.eig(): Computes the eigenvalues and eigenvectors of a matrix.
# np.linalg.svd(): Computes the singular value decomposition of a matrix.
print(alpha)
# print(residuals)
# print(rank)
# print(s)
 
#plot the results
plt.figure(figsize = (10,8))
plt.plot(X,Y,'b.')
plt.plot(X,alpha[0]*X + alpha[1],'r')
plt.xlabel('X')
plt.ylabel('Y')
plt.show() 
