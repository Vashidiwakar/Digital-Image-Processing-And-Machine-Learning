import numpy as np
import matplotlib.pyplot as plt
import cv2

#Loading the image 
image = cv2.imread('C:\\Users\\sachi\\OneDrive\\Desktop\\Machine_learning\\Train.jpg')
image1 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.subplot(1,2,1)
plt.imshow(image1)
plt.title("image1")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(image2)
plt.title("image2")
plt.axis("off")
plt.show()
