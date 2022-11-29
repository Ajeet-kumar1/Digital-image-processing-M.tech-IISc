# Import necessary libraries and function files
import numpy as np
import matplotlib.pyplot as plt
from full_scale_streching import streching


# Define a function to create sinusoid image and compute thier DFT
def sinusoid_image(u0, v0):
    M = N = 501                                                             # Dimension of image      
    m = np.array([np.linspace(0, M-1, M)])                                  # Create an array of length M having separation of index a
    n = m.T                                                                 # To treat it as another axis take tranpose of it
    output_image = np.sin(2 * np.pi * (m*u0 + n*v0) / N)                    # Apply formula
    dft_of_image = np.fft.fft2(output_image)                                # Comput it DFT


    return output_image, dft_of_image


################################################################# Driver Code #########################

if __name__ == '__main__':
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
    plt.show()


