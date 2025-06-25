import numpy as np
import cv2
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