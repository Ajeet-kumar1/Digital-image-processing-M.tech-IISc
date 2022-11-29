import numpy as np
import cv2
import matplotlib.pyplot as plt

################################################# twiddle factor and DFT calculation ###########################################

def DFT_calculator(image):
    N , N = image.shape                                                   # Image should be of square shape
    m = np.array([np.linspace(0, N-1, N)])                                # Create an array of length M having separation of index one
    n = m.T
    indexed = m*n                                                         # Two dimensional index matrix, for twiddle matrix
    twiddle_matrix = np.exp((-1j*2*np.pi*indexed)/N)                      # Matrix A (Given in formula)
    twiddle_hermitian = np.ndarray.conjugate(twiddle_matrix)              # A transpose matrix
    
    dummy_output = np.dot(twiddle_hermitian, image)
    final_output = np.dot(dummy_output, twiddle_matrix)
    final_output = np.flip(final_output)                                  # Flip your image to get correct orientation                  
    final_output = np.fliplr(final_output)

    return final_output, twiddle_matrix, twiddle_hermitian

################################################ Driver code ###################################################################


if __name__ == '__main__':
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