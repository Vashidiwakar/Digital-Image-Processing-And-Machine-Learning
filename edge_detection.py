import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def apply_sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    return sobel

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
# Apply the edge detection operators
sobel = apply_sobel(image)
prewitt = apply_prewitt(image)
roberts = apply_roberts(image)

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(prewitt, cmap='gray')
plt.title('Prewitt')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(roberts, cmap='gray')
plt.title('Roberts')
plt.axis('off')

plt.show()