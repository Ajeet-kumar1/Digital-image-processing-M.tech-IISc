import numpy as np
import cv2
import matplotlib.pyplot as plt
import edge_detection_prewitt
from edge_detection_prewitt import edge_detect                                    # Import previously made function



############################################################ Driver code #####################################################################
if __name__=='__main__':

    kernel = edge_detection_prewitt.gaussiun_kernel(5, 0, 1)
    kernel_x = np.array([
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

    ##################################################################### Rotate the image ##################################################################
    checkerboard_image = cv2.rotate(checkerboard_image, cv2.ROTATE_90_CLOCKWISE)
    noisy_checker_image = cv2.rotate(noisy_checker_image, cv2.ROTATE_90_CLOCKWISE)
    coin_image = cv2.rotate(coin_image, cv2.ROTATE_90_CLOCKWISE)
    noisy_coin = cv2.rotate(noisy_coin, cv2.ROTATE_90_CLOCKWISE)

    
    #################################################################### Plot the original image ############################################################

    plt.subplot(2, 4, 1)
    plt.imshow(checkerboard_image, cmap='gray')
    plt.title('Rotated image')

    plt.subplot(2, 4, 2)
    plt.imshow(noisy_checker_image, cmap='gray')
    
    plt.subplot(2, 4, 3)
    plt.imshow(coin_image, cmap='gray')
    
    plt.subplot(2, 4, 4)
    plt.imshow(noisy_coin, cmap='gray')
    
    ################################################################### Now plot the edges of image #########################################################

    edge_checkerboard = edge_detect(checkerboard_image, kernel_x, kernel_y)
    plt.subplot(2, 4, 5)
    plt.imshow(edge_checkerboard, cmap='gray')
    plt.title('Edges of image after rotation')

    edge_noisy_checker = edge_detect(noisy_checker_image, kernel_x, kernel_y)
    plt.subplot(2, 4, 6)
    plt.imshow(edge_noisy_checker, cmap='gray')

    edge_coin = edge_detect(coin_image, kernel_x, kernel_y)
    plt.subplot(2, 4, 7)
    plt.imshow(edge_coin, cmap='gray')

    edge_noisy = edge_detect(noisy_coin, kernel_x, kernel_y)
    plt.subplot(2, 4, 8)
    plt.imshow(edge_noisy, cmap='gray')
    plt.show()