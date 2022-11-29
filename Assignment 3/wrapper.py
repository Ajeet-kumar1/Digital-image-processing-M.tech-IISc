

from sinusoidal import*
from filtering import*
from gaussianLPF import*
from image_denoising import*
from bilateral_filter import*
from DFT_matrix import*





i = input("Press Enter in terminal through keyboard to run the program 1 (a): ")
# Call the function to from image and compute DFT 
imageA, imageA_dft = sinusoid_image(u0 = 40, v0 = 60)
imageB, imageB_dft = sinusoid_image(u0 = 20, v0 = 100)

# Evaluate absolute value, take log and do streching
abslute_A = np.abs(imageA_dft)
log_A = np.log(1 + abslute_A)
streched_A = streching(log_A)

abslute_B = np.abs(imageB_dft)
log_B = np.log(1 + abslute_B)
streched_B = streching(log_B)

# Now plot the images A
plt.figure(1)
plt.subplot(1, 2, 1)
plt.title('Image A')
plt.imshow(imageA, cmap="gray")
plt.subplot(1,2,2)
plt.imshow(streched_A, cmap='gray')
plt.title('DFT of Image A')

# Do the same for image B
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(imageB, cmap="gray")
plt.title('Image B')
plt.subplot(1,2,2)
plt.imshow(log_B, cmap='gray')
plt.title('DFT of Image B')

################################################ Add both DFTs and find IDFT ##############
final_DFT = imageA_dft + imageB_dft
final_IDFT_image = np.fft.ifft2(final_DFT)

################################################ Plot the IDFT image ######################
plt.figure(3)
plt.subplot(1,2,1)
abs_final_IDFT_image = np.abs(final_IDFT_image)
plt.imshow(abs_final_IDFT_image, cmap='gray')
plt.title('IDFT of sum in frequency domain')
################################################ Point wise addition of images ################
final_image = imageA + imageB
plt.subplot(1, 2, 2)
plt.imshow(final_image, cmap='gray')
plt.title('Point wise Sum in spatial domain')
print('After watching the output close all picture window in order to proceed for next program')
plt.show()

#########################################################################################################################################

i = input("Press Enter to run the program 1 (b): ")
###########################################################################################################################################


##################################################### Driver code ###############################################

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
print('After watching the output close all picture window in order to proceed for next program')
plt.show()

#################################################################################################################################################
i = input('Press "Enter" to run the program 1(c)')
###################################################### Driver code ########################################################

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
print('After watching the output close all picture window in order to proceed for next program')
plt.show()

#################################################################################################################################################
i = input('Press "Enter" to run the program 2(a)')


######################################################## Driver code #############################################################################



original_image = cv2.imread('Images\circuitboard.tif', 0)                       # Read the image


filtered_image, median_conv = convolution(original_image, 3)                    # Call the function and get the output

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
print('After watching the output close all picture window in order to proceed for next program')
plt.show()

#################################################################################################################################################
i = input('Press "Enter" to run the program 2(b)')


original_image = cv2.imread('book.png',0)                                  # Read the image
out_img, Gmap, Hmap, P = bilateral_filter(original_image)                  # Apply bilateral filter on it 
gaussian_smooth = cv2.GaussianBlur(original_image,(7,7),0)                 # Apply inbuilt gaussian smoothing for comparison 

# Plot the original image
plt.figure(1)                                                              
plt.subplot(1,3,1)
plt.imshow(original_image, cmap='gray')
plt.title('Original image')

# Plot the gaussian smoothed image for comparison
plt.subplot(1, 3, 2)
plt.imshow(gaussian_smooth, cmap='gray')
plt.title('Gaussian smooth image')

# Plot the output image
plt.subplot(1, 3, 3)
plt.imshow(out_img, cmap='gray')
plt.title('Output of bilateral filter')

# Plot the patch centered at (178, 260)

plt.figure(2)
plt.subplot(1, 4, 1)
plt.imshow(P, cmap='gray')
plt.title('Patch')

plt.subplot(1, 4, 2)
plt.imshow(Gmap, cmap='gray')
plt.title('Gmap')

plt.subplot(1, 4, 3)
plt.imshow(Hmap, cmap='gray')
plt.title("Hmap")

plt.subplot(1, 4, 4)
plt.imshow(np.dot(Gmap, Hmap), cmap='gray')
plt.title('Gmap * Hmap')
plt.show()

#################################################################################################################################################
i = input('Press "Enter" to run the program 3')

################################################ Driver code ###################################################################



image = cv2.imread('Images\characters.tif',0)                                      # Read the image
final_output, twiddle, twiddle_transpose = DFT_calculator(image)      # Calculate DFT by self made function


dftmtx = np.fft.fft2(image)                                           # Calculate Dft by inbuilt function for comparison


# Plot the output of manual coded DFT
plt.subplot(1, 3, 1)
plt.imshow(np.log(1 + np.abs(final_output)), cmap='gray')
plt.title('DFT by self coded function')

# Plot the DFT by inbuilt function
plt.subplot(1, 3, 2)
plt.imshow(np.log(1 + np.abs(dftmtx)), cmap='gray')
plt.title('DFT by inbuilt function')

# Plot A.T * A 
plt.subplot(1,3,3)
plt.imshow(abs(twiddle_transpose * twiddle), cmap = 'gray')
plt.title('A_transpose * A')

# Print mean square error using inbuilt function
print('Mean square error is :', np.abs(np.square(dftmtx - final_output)).mean(axis=None))

#Plot original image vs inverse FFT image for comparison
plt.figure(2)
    
# Plot original image
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original image')

idf = np.fft.ifft2(final_output)
plt.subplot(1,2,2)
plt.imshow(abs(idf), cmap='gray')
plt.title('Inverse DFT of manually coded DFT of image')
plt.show()
