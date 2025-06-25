import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_laplacian(image):
    # Step 1: Noise Reduction with Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Step 2: Apply Laplacian Operator
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    
    # Step 3: Convert to Absolute Values
    laplacian = np.abs(laplacian)
    
    # Step 4: Normalize to 8-bit Range
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    laplacian = np.uint8(laplacian)
    
    return laplacian

# Example usage:
image = cv2.imread('Train.jpg', cv2.IMREAD_GRAYSCALE)
edges = apply_laplacian(image)
plt.imshow(edges)
plt.title("Output_image")
plt.axis("off")
plt.show()
# cv2.imshow('Laplacian Edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

