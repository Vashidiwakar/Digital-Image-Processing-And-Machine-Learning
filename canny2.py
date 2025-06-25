import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Train.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur to smooth the image
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# Perform Canny Edge Detection
edges = cv2.Canny(blurred_image, 50, 150)

# Display the original image and the edges
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Canny Edges'), plt.axis('off')
plt.show()
