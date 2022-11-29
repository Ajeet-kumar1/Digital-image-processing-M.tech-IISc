
import numpy as np
import cv2
import matplotlib.pyplot as plt




############################################################ Bilateral Filter ################################################################
def bilateral_filter(original_image):
    row, col = original_image.shape                     
    output_img = np.zeros([row, col])                                          # A matrix of zeros to store the output
    m = np.arange(row)                                                         # For patch indexing , I(m, n)
    n = np.arange(col)
    [I, J] = np.meshgrid(m, n, indexing='ij')                                  # Create 2D indexed for gaussian calculation 
    variance_Gauss = 10
    variance_luminance = 0.1
    M = 3                                                                      # To make window size 7x7
    kg = 10 
    kh = 20                                                                    
    for i  in range(row):
        for j in range(col):
            patch = original_image[i-M:i+M, j-M:j+M]                           # Make a patch and store in a variable patch
            patch_index_x = I[i-M:i+M, j-M:j+M]                                # Find the index of all element of patch.
            patch_index_y = J[i-M:i+M, j-M:j+M]
            
            # Find guassian weight by applying formula
            gauss_weight = kg * np.exp(-(np.square(i - patch_index_x) + np.square(j - patch_index_y))/ 2* variance_Gauss)

            # And also apply formula in luminance distance
            luninance_distance = kh * np.exp(-((original_image[i][j] -  patch)**2)/ 2* variance_luminance) 

            # Find the coefficient Kij
            Kij = np.sum(gauss_weight*luninance_distance)

            # Finally get the vlue of output pixel
            output_img[i][j] = (np.sum(patch*gauss_weight*luninance_distance))/Kij
            if i==178 and j ==260:
                Gmap = gauss_weight
                Hmap = luninance_distance
                P = patch
    return output_img, Gmap, Hmap, P

    

            
        
if __name__=='__main__':
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