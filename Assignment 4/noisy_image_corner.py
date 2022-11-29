# Import all the requirement
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from harris_corner import harris_corner

############################################### Define the function which rotate scale and add noise in image #########################################
def rotate_scale_noise(image):
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)                                                 # Rotate the image
    image = cv2.resize(image, (500, 500))                                                              # scale the image
                                                                   
    P, Q = image.shape                                                                                 # Now add the noise. Salt and pepper noise is apopular
    number_of_pixels = random.randint(100, 5000)                                                       # So I am adding that noise
    for i in range(number_of_pixels):                        
        x_index = random.randint(0, P - 1)                                                             # choose a pixel randomly 
        y_index = random.randint(0, Q - 1)
        image[x_index][y_index] = 255                                                                  # And make it 255  graylevel (Addition of salt)
    
    for j in range(number_of_pixels):                                                                
        x_index = random.randint(0, P - 1)                                                             # Choose a pixel randomly
        y_index = random.randint(0, Q - 1)
        image[x_index][y_index] = 0                                                                    # Make it dark (addition of pepper)

    return image
    

############################################################Driver code ######################################################################################
if __name__=='__main__':
    # Read the both test images
    checkerboard = cv2.imread('Images\Checkerboard.png', 0)                                           
    mainbuilding = cv2.imread('Images\MainBuilding.png', 0)
    
    # Apply rotate scale and noise addition on image                                                 
    n_checkerboard = rotate_scale_noise(checkerboard)
    n_mainbuilding = rotate_scale_noise(mainbuilding)


    # Detect the corner in modified image
    corners_checker = harris_corner(image=n_checkerboard, thresold=10,sensitivity=0.05,sigma=2)
    corners_mainb = harris_corner(image=n_mainbuilding,thresold=50,sensitivity=0.05,sigma=0.2)

    # Plot the checkerboard original image 
    plt.subplot(2, 3, 1)
    plt.imshow(checkerboard, cmap='gray')
    plt.title('original image')
    
    # Plot noisy image
    plt.subplot(2, 3, 2)
    plt.imshow(n_checkerboard, cmap='gray')
    plt.title('Noisy roted scaled image')

    # Plot Corners of noisy rotated image
    plt.subplot(2, 3, 3)
    plt.imshow(corners_checker, cmap='gray')
    plt.title('Corner of Noisy roted scaled image')
    
    # Plot mainbuilding image
    plt.subplot(2, 3, 4)
    plt.imshow(mainbuilding, cmap='gray')
    plt.title('original image')

    plt.subplot(2, 3, 5)
    plt.imshow(n_mainbuilding, cmap='gray')
    plt.title('Noisy roted scaled image')

    plt.subplot(2, 3, 6)
    plt.imshow(corners_mainb, cmap='gray')
    plt.title('Corner of Noisy roted scaled image')
    plt.show()
 