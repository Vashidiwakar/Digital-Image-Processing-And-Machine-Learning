import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def apply_roberts(image):
    kernelx = np.array([[1, 0], [0, -1]], dtype=int)
    kernely = np.array([[0, 1], [-1, 0]], dtype=int)
    # Apply the kernels to the grayscale image
    roberts_x = cv2.filter2D(image, cv2.CV_64F, kernelx)
    roberts_y = cv2.filter2D(image, cv2.CV_64F, kernely)
    
    # Convert to absolute values and then to float32
    roberts_x = np.abs(roberts_x).astype(np.float32)
    roberts_y = np.abs(roberts_y).astype(np.float32)
    roberts = cv2.magnitude(roberts_x, roberts_y)
    return roberts

# Read and convert the image to grayscale
img = data.coffee()
image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

roberts = apply_roberts(image)
# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(roberts, cmap='gray')
plt.title('Roberts')
plt.axis('off')

plt.show()

