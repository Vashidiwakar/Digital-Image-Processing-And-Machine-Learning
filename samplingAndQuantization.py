import numpy as np
import matplotlib.pyplot as plt
from skimage import io,color
from skimage.transform import resize
import cv2 

#Function for downsampling
def downsample(image,factor):
  new_h = image.shape[0]//factor
  new_w = image.shape[1]//factor
  downsampled_image = resize(image,(new_h,new_w),mode = "constant",anti_aliasing = False)
  return downsampled_image

#Function for upsampling
def upsample(image,factor):
  new_h = image.shape[0]*factor
  new_w = image.shape[1]*factor
  upsampled_image = resize(image,(new_h,new_w),mode = "constant",anti_aliasing = False)
  return upsampled_image

#Function for quantization
def quantization(image,num_levels): 
  #check whether the image is RGB, if it is then convert it into grayscale
  if(image.ndim==3):
    Grayscale_image = color.rgb2gray(image)
    quantization_levels = np.linspace(0,1,num_levels)
  quantized_image = np.digitize(Grayscale_image,quantization_levels) - 1
  quantized_image = quantized_image / (num_levels - 1)  # Scale to [0, 1] range
  return quantized_image

original_image = cv2.imread('Train.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10,6))
# plt.subplot(1,5,1)
# plt.imshow(original_image)
# plt.title("ORIGINAL_IMAGE")
# plt.axis("off")


# DOWNSAMPLED IMAGES
img1 = downsample(original_image,4)
plt.subplot(1,5,2)
plt.imshow(img1)
plt.title("4X")
plt.axis("off")

img2 = downsample(original_image,8)
plt.subplot(1,5,3)
plt.imshow(img2)
plt.title("8X")
plt.axis("off")

img3 = downsample(original_image,16)
plt.subplot(1,5,4)
plt.imshow(img3)
plt.title("16X")
plt.axis("off")

img4 = downsample(original_image,32)
plt.subplot(1,5,5)
plt.imshow(img4)
plt.title("32X")
plt.axis("off")
plt.show()


#UPSAMPLED IMAGES
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.imshow(original_image)
# plt.title("ORIGINAL_IMAGE")
# plt.axis("off")

# img1 = upsample(original_image,8)
# plt.subplot(1,2,2)
# plt.imshow(img1)
# plt.title("8X")
# plt.axis("off")

# plt.show()
#QUANTIZED IMAGES
# plt.figure(figsize=(10,6))
# plt.subplot(1,5,1)
# plt.imshow(original_image)
# plt.title("ORIGINAL_IMAGE")
# plt.axis("off")

# img1 = quantization(original_image,4)
# plt.subplot(1,5,2)
# plt.imshow(img1,cmap = "gray")
# plt.title("4X")
# plt.axis("off")

# img2 = quantization(original_image,8)
# plt.subplot(1,5,3)
# plt.imshow(img2,cmap = "gray")
# plt.title("8X")
# plt.axis("off")

# img3 = quantization(original_image,16)
# plt.subplot(1,5,4)
# plt.imshow(img3,cmap = "gray")
# plt.title("16X")
# plt.axis("off")

# img4 = quantization(original_image,32)
# plt.subplot(1,5,5)
# plt.imshow(img4,cmap = "gray")
# plt.title("32X")
# plt.axis("off")
# plt.show()
