# Import requirements
import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_smoothing import smoothing
from filtered_downsampling import gaussiun_kernel

# Define a function convolution
def convolution(image, kernel):
    windows = np.lib.stride_tricks.sliding_window_view(image, kernel.shape)
    out = np.einsum('ij,klij->kl',kernel, windows)
    return out

# Define a function to detect edges
def edge_detect(image, kernel_x, kernel_y):
    out_x = convolution(image, kernel_x)                                                                # Take differentiation in x-axis and y-axis
    out_y = convolution(image, kernel_y)
    output = np.sqrt(out_x**2 + out_y**2)                                                               # Square and add them 
    
    output = (output > 100) * output                                                                    # Do the thresolding  
    return output


########################################################### Driver code ########################################################################
if __name__=='__main__':
    kernel = gaussiun_kernel(5, 0, 1)                                                                   # Evaluate the gaussian kernel
    kernel_x = np.array([                                                                               # Define x-axis and y-axis differentiator
                        [-1,-1,-1],
                        [0, 0, 0],
                        [1, 1, 1]
                                ])

    kernel_y = np.array([
                        [-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]
                                ])

    checkerboard_image = cv2.imread('Images\Checkerboard.png', 0)                      # Read all images   
    noisy_checker_image = cv2.imread('Images\oisyCheckerboard.png', 0)
    coin_image = cv2.imread('Images\Coins.png', 0)
    noisy_coin = cv2.imread('Images\oisyCoins.png', 0)
    
    #################################################################### Plot the original image ############################################################

    plt.subplot(2, 4, 1)
    plt.imshow(checkerboard_image, cmap='gray')
    plt.title('Original image')

    plt.subplot(2, 4, 2)
    plt.imshow(noisy_checker_image, cmap='gray')
    plt.title('Original image')
    
    plt.subplot(2, 4, 3)
    plt.imshow(coin_image, cmap='gray')
    plt.title('Original image')
    
    plt.subplot(2, 4, 4)
    plt.imshow(noisy_coin, cmap='gray')
    plt.title('Original image')
    
    ################################################################### Now plot the edges of image #########################################################

    edge_checkerboard = edge_detect(checkerboard_image, kernel_x, kernel_y)
    plt.subplot(2, 4, 5)
    plt.imshow(edge_checkerboard, cmap='gray')
    plt.title('Edges of image without filtering')

    edge_noisy_checker = edge_detect(noisy_checker_image, kernel_x, kernel_y)
    plt.subplot(2, 4, 6)
    plt.imshow(edge_noisy_checker, cmap='gray')
    plt.title('Edges of image without filtering')

    edge_coin = edge_detect(coin_image, kernel_x, kernel_y)
    plt.subplot(2, 4, 7)
    plt.imshow(edge_coin, cmap='gray')
    plt.title('Edges of image without filtering')

    edge_noisy = edge_detect(noisy_coin, kernel_x, kernel_y)
    plt.subplot(2, 4, 8)
    plt.imshow(edge_noisy, cmap='gray')
    plt.title('Edges of image without filtering')

    #################################################################Apply gaussian Smoothing###### ######################################################
    checkerboard_smooth = smoothing(checkerboard_image, kernel)
    noisy_checker_smooth = smoothing(noisy_checker_image, kernel ) 
    coin_smooth = smoothing(coin_image,  kernel)
    noisy_coin_smooth = smoothing(noisy_coin, kernel)

    ################################################################## Plot the smoothen image ###############################################################
    plt.figure(2)
    plt.subplot(2, 4, 1)
    plt.imshow(checkerboard_smooth, cmap='gray')
    plt.title('Smoothen image')

    plt.subplot(2, 4, 2)
    plt.imshow(noisy_checker_smooth, cmap='gray')
    plt.title('Smoothen image')
    
    plt.subplot(2, 4, 3)
    plt.imshow(coin_smooth, cmap='gray')
    plt.title('Smoothen image')
    
    plt.subplot(2, 4, 4)
    plt.imshow(noisy_coin_smooth, cmap='gray')
    plt.title('Smoothen image')
    
    ################################################################### Now plot the edges of smoothen image #########################################################

    edge_checkerboard = edge_detect(checkerboard_smooth, kernel_x, kernel_y)
    plt.subplot(2, 4, 5)
    plt.imshow(edge_checkerboard, cmap='gray')
    plt.title('Edges of image with smoothing')

    edge_noisy_checker = edge_detect(noisy_checker_smooth, kernel_x, kernel_y)
    plt.subplot(2, 4, 6)
    plt.imshow(edge_noisy_checker, cmap='gray')
    plt.title('Edges of image with smoothing')

    edge_coin = edge_detect(coin_smooth, kernel_x, kernel_y)
    plt.subplot(2, 4, 7)
    plt.imshow(edge_coin, cmap='gray')
    plt.title('Edges of image with smoothing')

    edge_noisy = edge_detect(noisy_coin_smooth, kernel_x, kernel_y)
    plt.subplot(2, 4, 8)
    plt.imshow(edge_noisy, cmap='gray')
    plt.title('Edges of image with smoothing')
    plt.show()

