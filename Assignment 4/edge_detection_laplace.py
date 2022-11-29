
import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_smoothing import smoothing
from filtered_downsampling import gaussiun_kernel

def edge_dection(image, kernel):
    windows = np.lib.stride_tricks.sliding_window_view(image, kernel.shape)
    out = np.einsum('ij,klij->kl',kernel, windows)
    return out




####################################################################### Driver code ############################################################
if __name__=='__main__':
    # Define laplacian operator
    laplace_kernel = np.array([
                        [0,-1, 0],
                        [-1,4,-1],
                        [0, -1,0]
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
    
    ################################################################### Now plot the edges of image ####################################################

    edge_checkerboard = edge_dection(checkerboard_image, laplace_kernel)
    plt.subplot(2, 4, 5)
    plt.imshow(edge_checkerboard, cmap='gray')
    plt.title('Edges of image')
    
    noisy_checker_smooth = smoothing(noisy_checker_image, gaussiun_kernel(5, 0, 1) )                           # For noisy first smooth
    edge_noisy_checker = edge_dection(noisy_checker_smooth, laplace_kernel)
    edge_noisy_checker = (edge_noisy_checker > 5) * edge_noisy_checker                                       # Threshold in case on noisy image
    plt.subplot(2, 4, 6)
    plt.imshow(edge_noisy_checker, cmap='gray')
    plt.title('Edges of image after thresold= 5')

    edge_coin = edge_dection(coin_image, laplace_kernel)
    plt.subplot(2, 4, 7)
    plt.imshow(edge_coin, cmap='gray')
    plt.title('Edges of image')
 
    noisy_coin = smoothing(noisy_coin, gaussiun_kernel(5, 0, 1) )
    edge_noisy = edge_dection(noisy_coin, laplace_kernel)
    plt.subplot(2, 4, 8)
    edge_noisy = (edge_noisy > 15) * edge_noisy                                                              # Threshold noisy image 
    plt.imshow(edge_noisy, cmap='gray')
    plt.title('Edges of image after thresold = 15')
    plt.show()

    