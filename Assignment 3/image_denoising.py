import numpy as np
import cv2
import matplotlib.pyplot as plt

########################################################### Mean filter and median ######################################################

# Do the convolution on noisy image
def convolution(original_image, kernel_shape):
    k = kernel_shape
    
    # Determine the output shape of convolved image
    output_size_row = original_image.shape[0] - (k - 1)
    output_size_col = original_image.shape[1] - (k - 1)
    convolved_img = np.zeros(shape=(output_size_row, output_size_col))              # Make an output shape zero's image for mean
    convolved_img_median = np.zeros(shape=(output_size_row, output_size_col))       # Make an output shape zero's image for median 

    
    # Iterate over the rows
    for i in range(output_size_row):
        # Iterate over the columns
        for j in range(output_size_col):
            patch = original_image[i:i+k, j:j+k]                                    # Break image into kernel size patches
            patch_median = np.median(patch)                                         # Evaluate median of a patch
            
            convolved_img[i, j] = np.sum(patch)/k*k                                 # Just take the average of all nine pixel
            #convolved_img_median[i:i+k, j:j+k].fill(patch_median)                   # Assign all zeros to median value in corresponding patch
            convolved_img_median[i:i+k, j:j+k].fill(patch_median)

        
    return convolved_img, convolved_img_median

######################################################## Driver code #############################################################################

if __name__ == '__main__':

    original_image = cv2.imread('Images\circuitboard.tif', 0)                       # Read the image

    
    filtered_image, median_conv = convolution(original_image, 3)    # Call the function and get the output

    plt.figure(1)
    plt.subplot(1,3,1)                                                              # Plot original image
    plt.imshow(original_image, cmap= 'gray')
    plt.title('Original image')

    plt.subplot(1,3,2)                                                              # Plot mean filtered image
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Mean filtered image')

    plt.subplot(1,3,3)                                                              # Median filtered image
    plt.imshow(median_conv, cmap='gray')
    plt.title('Meadian filtered image')
    plt.show()