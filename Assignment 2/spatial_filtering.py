import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add padding in image in order to get output of same size
def add_padding(img, kernel_size):
    # The below formula for padding width is valid only for stride value 1 and square kernel matrix.
    padding_height = int((kernel_size-1) / 2)
    padding_width = int((kernel_size-1) / 2)

    # Create a zeros matrix having shape of padded image
    img_with_padding = np.zeros(shape=(
        img.shape[0] + padding_height * 2,  # Multiply with two because we need padding on top and bottom
        img.shape[1] + padding_width * 2
    ))
    # Now replace zero values with image values other than padding
    img_with_padding[padding_width:-padding_width, padding_width:-padding_width] = img
    return img_with_padding

# Do the convolution on padded image
def convolution(original_image, padded_image, kernel):
    # Take a dummy variable for multiplication of filter length
    k = kernel.shape[0]
    
    # Make a zero's matrix of orginal image size to store the output
    output_size = original_image.shape[0]
    convolved_img = np.zeros(shape=(output_size, output_size))
    
    # Iterate over the rows
    for i in range(output_size):
        # Iterate over the columns
        for j in range(output_size):
            # Fetch the filter size patch from padded image 
            patch = padded_image[i:i+k, j:j+k]
            # Use formula of convolution(element-wise multiplication and summation of the result) + Store the result in the convolved_img at corresponding location
            convolved_img[i, j] = np.sum(np.multiply(patch, kernel))
        
    return convolved_img

############################################# Driver code ############################
if __name__=='__main__':
    sharpen = np.array([    # Laplacian sharpen filter
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0] 
    ])
    # Read the Blur image and convert into gray // Question 3 (a)
    img_blur = cv2.imread('images\Blur.png')
    img_blur_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    #Now the add padding and do convolution
    padded_image_blur = add_padding(img_blur_gray, kernel_size=3)
    convolved_image_blur = convolution(img_blur_gray, padded_image_blur, kernel=sharpen)
    # Now add sharpen image in blur image with coefficient multiplication 
    add_blur = np.add(convolved_image_blur*4.5, img_blur_gray) # Scaling factor is 2.5

    # Now plot original and sharpened image
    plt.subplot(1,2,1)
    plt.imshow(img_blur_gray, cmap='gray', vmin=0, vmax=255)
    plt.title('Original image Blur.png')

    plt.subplot(1,2,2)
    plt.imshow(add_blur, cmap='gray', vmin=0, vmax=255)
    plt.title('Sharpend image of Blur.png')



###################### Do the same with noisy.png in Question 3 (b)

#    Read the image and convert into gray scale 
    img_noisy = cv2.imread('images\oisy.png')
    img_noisy_gray = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)

    # Do padding and convolution
    padded_image_noisy = add_padding(img_noisy_gray, kernel_size=3)
    convolved_image_noisy = convolution(img_noisy_gray, padded_image_noisy, kernel=sharpen)
    # Now add images with same scaling factor
    add_noisy = np.add(convolved_image_noisy*2.5, img_noisy_gray)
    
    # Plot the both images
    plt.figure(2)
    plt.subplot(1,2,1)
    plt.imshow(img_noisy_gray, cmap='gray', vmin=0, vmax=255)
    plt.title('Original image Noisy.png')

    plt.subplot(1,2,2)
    plt.imshow(add_noisy, cmap='gray', vmin=0, vmax=255)
    plt.title('Sharpend image of Noisy.png')
    plt.show()