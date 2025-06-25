import numpy as np
import matplotlib.pyplot as plt
from skimage import io,data
from sklearn.cluster import KMeans
from PIL import Image

#importing image
image = data.coffee()

#converting image into array
img = np.array(image)
#converting into 2D array
pixels = img.reshape((-1,3))
# print(image_np.shape)

kmeans = KMeans(n_clusters = 16,random_state = 42)
kmeans.fit(pixels)

# Replace each pixel with the color of its cluster's centroid
new_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(int)
quantized_image_np = new_pixels.reshape(img.shape)
#converting array into image
quantized_image = Image.fromarray(quantized_image_np.astype('uint8'), 'RGB')

plt.subplot(121)
plt.imshow(image)
plt.title("Original_image")
plt.axis("off")

plt.subplot(122)
plt.imshow(quantized_image)
plt.title("Qunatized_image")
plt.axis("off")

plt.show()