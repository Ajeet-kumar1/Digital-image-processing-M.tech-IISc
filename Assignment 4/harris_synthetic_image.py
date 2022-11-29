import cv2
import numpy as np
from matplotlib import pyplot as plt
from harris_corner import harris_corner

################################################################# Define a functio to create the synthetic image ########################################
def synthetic_image():
    dummy_image = np.zeros((1000, 1000, 3), dtype = "uint8")                                                  # Create a dummy image 
    cv2.line(dummy_image, (100, 50), (900, 50), (255, 0, 0), 5)                                               # Draw a line
    cv2.rectangle(dummy_image, (50, 200), (400, 500), (0, 255, 0), cv2.FILLED)                                # Draw a rectangle
    cv2.rectangle(dummy_image, (800, 100), (850, 150), (255, 255, 0), cv2.FILLED)                             # Draw another rectangle
    cv2.circle(dummy_image, (700, 340), 150, (255, 0, 0), cv2.FILLED)                                         # Draw circle
    pts = [(200, 600), (300, 980), (900, 800)] 
    cv2.fillPoly(dummy_image, np.array([pts]), (255, 0, 0))                                                   # Draw triangle

    return dummy_image



if __name__=='__main__':
    synthetic_image1 = synthetic_image()
    synthetic_gray = cv2.cvtColor(synthetic_image1, cv2.COLOR_RGB2GRAY)
    corners = harris_corner(image=synthetic_gray,thresold=10,sensitivity=0.05,sigma=1)

    plt.subplot(1, 2, 1)
    plt.imshow(synthetic_image1, cmap='gray')
    plt.title('Original synthetic image')

    plt.subplot(1, 2, 2)
    plt.imshow(corners, cmap='gray')
    plt.title('Corners of image')
    plt.show()
