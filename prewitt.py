import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def apply_prewitt(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    prewitt_x = cv2.filter2D(image, cv2.CV_64F, kernelx)
    prewitt_y = cv2.filter2D(image, cv2.CV_64F, kernely)
    
    # Convert to absolute values and then to float32
    prewitt_x = np.abs(prewitt_x).astype(np.float32)
    prewitt_y = np.abs(prewitt_y).astype(np.float32)
    prewitt = cv2.magnitude(prewitt_x, prewitt_y)
    return prewitt

# Read and convert the image to grayscale
img = data.coffee()
image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

prewitt = apply_prewitt(image)

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(prewitt, cmap='gray')
plt.title('Prewitt')
plt.axis('off')

plt.show()