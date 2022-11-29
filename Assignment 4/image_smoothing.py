import numpy as np
import matplotlib.pyplot as plt
import cv2
from filtered_downsampling import gaussiun_kernel                                        # From problem 1 (b) import gaussian kernel

# Make a convolution function
def smoothing(image, kernel):
    windows = np.lib.stride_tricks.sliding_window_view(image, kernel.shape)
    out = np.einsum('ij,klij->kl',kernel, windows)
    return out



if __name__=='__main__':
    kernel = gaussiun_kernel(5, 0, 5)                                                  # 5 X 5 kernel with variance 25
    
    checkerboard_image = cv2.imread('Images\Checkerboard.png', 0)                      # Read all images   
    noisy_checker_image = cv2.imread('Images\oisyCheckerboard.png', 0)
    coin_image = cv2.imread('Images\Coins.png', 0)
    noisy_coin = cv2.imread('Images\oisyCoins.png', 0)
    
    ####################################### Apply filtering on them ################################################################
    checkerboard_out = smoothing(checkerboard_image, kernel)
    blur1 = cv2.GaussianBlur(checkerboard_image,(5,5),0)                               # Use library function for comparison

    noisy_checker_out = smoothing(noisy_checker_image, kernel)
    blur2 = cv2.GaussianBlur(noisy_checker_image,(5,5),0)

    coin_out = smoothing(coin_image, kernel)
    blur3 = cv2.GaussianBlur(coin_image,(5,5),0)

    noisy_coin_out = smoothing(noisy_coin, kernel)
    blur4 = cv2.GaussianBlur(noisy_coin,(5,5),0)

    ############################################################# Plot them #########################################################


    plt.subplot(2, 4, 1)
    plt.imshow(checkerboard_out, cmap='gray')
    plt.title('Checker board output')

    plt.subplot(2, 4, 2)
    plt.imshow(noisy_checker_out, cmap='gray')
    plt.title('Noisy board output')

    plt.subplot(2, 4, 3)
    plt.imshow(coin_out, cmap='gray')
    plt.title('Coin image output')

    plt.subplot(2, 4, 4)
    plt.imshow(noisy_coin_out, cmap='gray')
    plt.title('Noisy coin output')

    ####################################################### Plot the libarary function output for comparison ##########################
    plt.subplot(2, 4, 5)
    plt.imshow(blur1, cmap='gray')
    plt.title('Checker board library output')

    plt.subplot(2, 4, 6)
    plt.imshow(blur2, cmap='gray')
    plt.title('Noisy checker libarary output')

    plt.subplot(2, 4, 7)
    plt.imshow(blur3, cmap='gray')
    plt.title('Coin image libaray output')

    plt.subplot(2, 4, 8)
    plt.imshow(blur4, cmap='gray')
    plt.title('Noisy coin library output')
    plt.show()