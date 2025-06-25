import numpy as np
import matplotlib.pyplot as plt

# Generate an example X array
X = np.linspace(0, 10, 100)

# Generate Y based on the expression
Y = 1 + X + X * np.random.random(len(X))

# Plotting the result
plt.scatter(X, Y, label='Y = 1 + X + X * random', alpha=0.5)
plt.plot(X, 1 + X, color='red', label='Y = 1 + X (without noise)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
