import numpy as np
import cv2
import matplotlib.pyplot as plt



############################### Ideal low pass filter #############################################################
def low_pass_filter(thresold, P, Q):
    all_black_image = np.zeros([P,Q])

    for u in range(P):
        for v in range(Q):
            dist = np.sqrt((u - P/2)**2 + (v - Q/2)**2)
            if dist > thresold:
                all_black_image[u][v] = 0
            else:
                all_black_image[u][v] = 1
    low_passed = all_black_image
    return low_passed


################################### Ideal high pass filter #######################################################

def high_pass_filter(thresold, P, Q):
    low_passed = low_pass_filter(thresold, P, Q)
    high_passed = 1 - low_passed
    return high_passed
########################################## Ideal Band pass filter ################################################
def band_pass_filter(Dh, Dl,P, Q):
    band_passed = high_pass_filter(Dh, P,Q) * low_pass_filter(Dl, P, Q)
    return band_passed


##################################################### Driver code ###############################################
if __name__=="__main__":
    ## Load the image find DFT of it
    dynamicSine_gray = cv2.imread('Images\dynamicSine.png', 0)                  # Read grayscale/ one layer of image
    P, Q = dynamicSine_gray.shape                                               # Find the image dimension for calculation purpose
    dft_of_image = np.fft.fft2(dynamicSine_gray)                                # Find the DFT of image
    dft_of_image_centred = np.fft.fftshift(dft_of_image)                        # Centered the frequency in the center of image


    ################################################### Apply low pass filter ###################################
    low_pass_filtr = low_pass_filter(20, P, Q)                                  # Low pass filter with cut off 20
    low_passed = np.multiply(dft_of_image_centred ,low_pass_filtr)              # Element wise multiplication of filter and image
    low_passed_uncenter = np.fft.ifftshift(low_passed)                          # Center shift
    inverse_fft = np.fft.ifft2(low_passed_uncenter)                             # Inverse DFT of low passed image
    plt.figure(1)
    plt.subplot(2,4,1)
    plt.imshow(low_pass_filtr, cmap='gray')                                     # Plot the filter
    plt.title('Low pass filter with D0 = 20')
    plt.subplot(2,4,5)
    plt.imshow(np.abs(inverse_fft), cmap='gray')                                # Plot filtered image
    plt.title('Low passed image with D0 = 20')


    ################################################## Apply high pass filter ##################################
    high_pass_filtr = high_pass_filter(60, P, Q)                                # High pass filter

    high_passed = np.multiply(dft_of_image_centred ,high_pass_filtr)            # Element wise multiplication of filter and image
    high_passed_uncenter = np.fft.ifftshift(high_passed)                        # Center shift
    inverse_fft = np.fft.ifft2(high_passed_uncenter)                            # Inverse DFT of low passed image
    plt.subplot(2,4,2)
    plt.imshow(high_pass_filtr, cmap='gray')                                    # Plot the filter
    plt.title('High pass filter with D0 = 60')
    plt.subplot(2,4,6)
    plt.imshow(np.abs(inverse_fft), cmap='gray')                                # Plot filtered image
    plt.title('High passed image with D0 = 60')

    ################################################### Band pass filter 1 ######################################
    band_pass_filtr1 = band_pass_filter(20, 40, P, Q)                           # Band pass filter

    band_passed1 = np.multiply(dft_of_image_centred ,band_pass_filtr1)          # Element wise multiplication of filter and image
    band_passed_uncenter1 = np.fft.ifftshift(band_passed1)                      # Center shift
    inverse_fft1 = np.fft.ifft2(band_passed_uncenter1)                          # Inverse DFT of low passed image
    plt.subplot(2,4,3)
    plt.imshow(band_pass_filtr1, cmap='gray')                                   # Plot the filter
    plt.title('Band pass filter with D0 = (20, 40)')
    plt.subplot(2,4,7)
    plt.imshow(np.abs(inverse_fft1), cmap='gray')                               # Plot filtered image
    plt.title('Band passed image with D0 = (20, 40)')

    
    #################################################### Band pass filter 2 ####################################
    band_pass_filtr2 = band_pass_filter(40, 60, P, Q)                            # Band pass filter

    band_passed2 = np.multiply(dft_of_image_centred ,band_pass_filtr2)           # Element wise multiplication of filter and image
    band_passed_uncenter2 = np.fft.ifftshift(band_passed2)                       # Center shift
    inverse_fft2 = np.fft.ifft2(band_passed_uncenter2)                           # Inverse DFT of low passed image
    plt.subplot(2,4,4)
    plt.imshow(band_pass_filtr2, cmap='gray')                                    # Plot the filter
    plt.title('Band pass filter with D0 = (40, 60)')
    plt.subplot(2,4,8)
    plt.imshow(np.abs(inverse_fft2), cmap='gray')                                # Plot filtered image
    plt.title('Band passed image with D0 = (40, 60)')
    plt.show()
    

