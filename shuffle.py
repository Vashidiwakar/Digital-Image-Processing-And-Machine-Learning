from sklearn.utils import shuffle
import numpy as np

# Example data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 0, 1, 0])

# Shuffling the data and labels together
X_shuffled, y_shuffled = shuffle(X, y, random_state=0)

print("Shuffled X:")
print(X_shuffled)
print("Shuffled y:")
print(y_shuffled)
