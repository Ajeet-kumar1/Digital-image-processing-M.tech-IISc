# Import all the requirement
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter


################################################################### Define the function for Harris corner and edge detection ################################

def harris_corner(image, thresold, sensitivity, sigma):                                                                       
    P, Q = image.shape

    sobel_x = np.array((                                                                      # Define the Sobel matrices  
                       [-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]), dtype="int32")
    sobel_y = np.array((
                       [-1,-2,-1],
                       [0, 0, 0],
                       [1, 2, 1]), dtype="int32")

    Ix = ndimage.convolve(image, sobel_x, mode='constant', cval=0.0)                         # Apply Sobel filter on the image with help of convolution
    Iy = ndimage.convolve(image, sobel_y, mode='constant', cval=0.0)

    Ixx = np.square(Ix)                                                                      # Find second order derivatives
    Iyy = np.square(Iy)
    Ixy = Ix * Iy                                                                            # Determine the cross correlation term

    Ixx = gaussian_filter(Ix**2, sigma)                                                    # Convolve these images with a larger Gaussian window
    Ixy = gaussian_filter(Iy*Ix, sigma)                                                    # Value of mean(sigma) is taken 2 for better output result(It 
    Iyy = gaussian_filter(Iy**2, sigma)                                                    # is just observation based)

    detM = Ixx * Iyy - Ixy **2                                                               # Calculate determinant and Trace of Matrix M
    traceM = Ixx + Iyy
    response_matrix = detM - sensitivity * traceM ** 2                                       # Now put the values in equation to get Matrix R 

    dummy_corners = np.zeros([P,Q])                                                          # Construct two matrix to store the output 
    #dummy_edges = dummy_corners

    for i in range(P):                                                                       # Iterate all the element of response matrix
        for j in range(Q):
            if response_matrix[i][j] > thresold:                                             # Compare with thresold vakue
                dummy_corners[i][j] =  255                                                   # Assign intensity value 255 at corner to make bright 
                   
            elif response_matrix[i][j] < thresold:                                           # Assign intensity value 255 at edges to make visible
                dummy_corners[i][j] = 0
                   
    return dummy_corners #, dummy_edges


############################################################################# Driver code ###################################################################

if __name__=='__main__':
    checkerboard = cv2.imread('Images\Checkerboard.png', 0)                                 # Read the both test images
    mainbuilding = cv2.imread('Images\MainBuilding.png', 0)

    img_corners_checker = harris_corner(image=checkerboard, thresold=10,sensitivity=0.05,sigma=2) # Call the function to detect the corner
    img_corners_mainb = harris_corner(image=mainbuilding,thresold=50,sensitivity=0.05,sigma=0.2) 

    plt.subplot(2, 2, 1)                                                                    # Plot the original image
    plt.imshow(checkerboard, cmap='gray')
    plt.title('Original image')
           
    plt.subplot(2, 2, 2)                                                                    # Plot the corner of image
    plt.imshow(img_corners_checker, cmap='gray')
    plt.title('Corner of checker board at thresold 0')
    
    plt.subplot(2, 2, 3)                                                                    # Plot the original image of main building
    plt.imshow(mainbuilding, cmap='gray')
    plt.title('Original image of main building')

    plt.subplot(2, 2, 4)                                                                    # Plot the corner of image
    plt.imshow(img_corners_mainb, cmap='gray')
    plt.title('Corner of main building at thresold 0')
    plt.show()

