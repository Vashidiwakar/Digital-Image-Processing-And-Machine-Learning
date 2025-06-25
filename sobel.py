import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def apply_sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    return sobel

img = data.coffee()
image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sobel = apply_sobel(image)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel')
plt.axis('off')

plt.show()
