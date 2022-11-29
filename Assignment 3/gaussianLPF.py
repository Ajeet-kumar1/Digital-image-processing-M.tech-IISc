import numpy as np
import cv2
import matplotlib.pyplot as plt
from filtering import low_pass_filter



################################## Gaussian low pass filter ##############################################################
def gaussian_low_pass_filter(thresold, P, Q):
    all_black_image = np.zeros([P,Q])                                              # Create an image having all pixel zero.
                                             
    for u in range(P):
        for v in range(Q):
            dist = np.sqrt((u - P/2)**2 + (v - Q/2)**2)                            # Itrate all the element and apply formula only
            gauss = np.exp(-(dist**2 )/ (2 * thresold**2))
            all_black_image[u][v] = gauss
    gaussian_lpf = all_black_image
    return gaussian_lpf

###################################################### Driver code ########################################################
if __name__=='__main__':
    original_image = cv2.imread('Images\characters.tif', 0)                       # Read the image find its DFT
    P, Q = original_image.shape
    dft_of_image = np.fft.fft2(original_image)                                    # Find the DFT of image
    dft_of_image_centred = np.fft.fftshift(dft_of_image)                          # Center the DFT
    ############################################ First ideal low pass filter ##############################################
    ideal_lpf = low_pass_filter(100, P, Q)
    ideal_lpf_image = np.multiply(dft_of_image_centred, ideal_lpf)                # Ideal low passed image
    ideal_lpf_image_uncent = np.fft.ifftshift(ideal_lpf_image)                    # Center shift
    inverse_fft = np.fft.ifft2(ideal_lpf_image_uncent)                            # Inverse DFT

    plt.figure(1)                                                                 # Plot the filter
    plt.subplot(2,2,1)
    plt.imshow(ideal_lpf, cmap = 'gray')
    plt.title('Ideal low pass filter with D0 = 100')

    plt.subplot(2, 2, 2)                                                          # Plot low passed image
    plt.imshow(abs(inverse_fft), cmap='gray')
    plt.title('Ideal low passed image')


    ############################################# Gaussian low pass filter ################################################

    gaussian_lpf = gaussian_low_pass_filter(100, P, Q)                             # Gaussian Low pass filter with cut off 100
    
    gauss_low_passed = np.multiply(dft_of_image_centred ,gaussian_lpf)             # Element wise multiplication of filter and image
    low_passed_uncenter = np.fft.ifftshift(gauss_low_passed)                       # Center shift
    inverse_fft = np.fft.ifft2(low_passed_uncenter)                                # Inverse DFT

    plt.subplot(2, 2, 3)                                                           # Plot the filter
    plt.imshow(gaussian_lpf, cmap='gray')
    plt.title('Gaussian low pass filter')

    plt.subplot(2,2,4)                                                             # Plot the filtered image
    plt.imshow(abs(inverse_fft), cmap='gray')
    plt.title('Gaussian low passed image')
    plt.show()