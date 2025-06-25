import numpy as np
import matplotlib.pyplot as plt
import cv2

#Loading the image 
image = cv2.imread('C:\\Users\\sachi\\OneDrive\\Desktop\\Machine_learning\\Train.jpg')
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def apply_canny(image, low_threshold, high_threshold):
    #Using Gaussian blur to smoothen the image -> noise reduction
    new_image = cv2.GaussianBlur(image,(3,3),1.4)

    #Finding gradient magnitude and gradient direction
    sobel_x = cv2.Sobel(new_image,cv2.CV_64F,1,0,ksize=3)
    sobel_y = cv2.Sobel(new_image,cv2.CV_64F,0,1,ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y,sobel_x)

    #Non-maximum suppression , double thresholding, edge tracking by hysteresis
    nms_image = cv2.Canny(new_image, low_threshold, high_threshold)
    
    return nms_image

image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny = apply_canny(image,50,90)
canny = cv2.cvtColor(canny,cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Grayscale_image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(canny)
plt.title("Output_image")
plt.axis("off")
plt.show()


