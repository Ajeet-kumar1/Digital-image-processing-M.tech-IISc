# Import required library
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from downsample import down_sample



# Define a function which evaluate the gaussian filter 
def gaussiun_kernel(size, sigma, std_deviation):
    indexing = size//2                                                                               # Perform integer division to indexing the filter
    t = np.linspace(-indexing, indexing, size)                                                       # Create a vector haing length equal to one dimension
    x, y = np.meshgrid(t, t)                                                                         # of filter
    sum_square = np.exp(- (x**2 + y**2)/ 2* std_deviation*std_deviation)                             # Now just apply the gaussian 2D formula
    guass = (1/(2 * np.pi * std_deviation * std_deviation)) * sum_square

    return guass

# Define a function for convolution of image with filter
def convolution(image, filter):
    return ndimage.convolve(image, filter, mode='constant', cval=0.0)


############################################################################## Driver code #################################################################
if __name__=='__main__':
    # Read the image
    image = cv2.imread('Images\city.png', 0)
    m, n = image.shape

    # Perform image smoothing
    kernel = gaussiun_kernel(5, 0, 2)
    convolved_img = convolution(image, kernel)

    # Plot the manually filtered image
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(convolved_img, cmap='gray')
    plt.title('Manually Filtered image')
    
    # Plot the libarary output for comparision 
    plt.subplot(1,2,2)
    blur = cv2.GaussianBlur(image,(5,5),2)
    plt.imshow(blur, cmap='gray')
    plt.title('Libarary filtered output')
    
    # Apply downsampling on manually filtered image
    down_img_2 = down_sample(convolved_img, 2)
    down_img_4 = down_sample(convolved_img, 4)
    down_img_5 = down_sample(convolved_img, 5)

    # Apply downsampling of library filtered image
    down_img_2_blur = down_sample(blur, 2)
    down_img_4_blur = down_sample(blur, 4)
    down_img_5_blur = down_sample(blur, 5)

    # Now plot the manually filtered downsampled image
    plt.figure(2)
    plt.subplot(2, 3, 1)
    plt.imshow(down_img_2, cmap='gray')
    plt.title('Downsampled by 2 manually filtered image')
    
    # Downsample by factor 4
    plt.subplot(2, 3, 2)
    plt.imshow(down_img_4, cmap='gray')
    plt.title('Downsampled by 4 manually filtered image')
    
    # Downsample by factor 5
    plt.subplot(2, 3, 3)
    plt.imshow(down_img_5, cmap='gray')
    plt.title('Downsampled by 5 manually filtered image')

    # Now plot the library filtered downsample image
    plt.subplot(2, 3, 4)
    plt.imshow(down_img_2_blur, cmap='gray')
    plt.title('Downsampled by 2 library filtered image')

    plt.subplot(2, 3, 5)
    plt.imshow(down_img_4_blur, cmap='gray')
    plt.title('Downsampled by 4 library filtered image')

    plt.subplot(2, 3, 6)
    plt.imshow(down_img_5_blur, cmap='gray')
    plt.title('Downsampled by 5 library filtered image')
    plt.show()




