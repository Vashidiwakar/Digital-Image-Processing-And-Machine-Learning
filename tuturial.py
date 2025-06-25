import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io , data
from PIL import Image

def quantize_image(img, k):
    # Load the image
    image = data.coffee()
    # plt.imshow()
    # plt.title("Original_image")
    # plt.axis("off")
    # plt.show()
    # img = image.convert('RGB')  # Ensure image is in RGB format
    image_np = np.array(img)
    
    # Reshape the image data to a 2D array of pixels
    pixels = image_np.reshape((-1, 3))
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    
    # Replace each pixel with the color of its cluster's centroid
    new_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(int)
    quantized_image_np = new_pixels.reshape(image_np.shape)
    
    # Convert the quantized array back to an image
    quantized_image = Image.fromarray(quantized_image_np.astype('uint8'), 'RGB')
    
    return quantized_image

# Parameters
img = data.coffee()
k = 16  # Number of clusters

# Perform image quantization
quantized_image = quantize_image(img, k)

# Display the original and quantized images
original_image = data.coffee()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Quantized Image with {k} colors')
plt.imshow(quantized_image)
plt.axis('off')

plt.show()

# Save the quantized image if needed
quantized_image.save('quantized_image.jpg')
