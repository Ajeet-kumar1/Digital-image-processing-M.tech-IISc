import numpy as np
import cv2
import matplotlib.pyplot as plt



# Define downsampling function
def down_sample(input_img, factor):
	m, n = input_img.shape
	down_img = np.zeros((m//factor, n//factor))                                # Create an output matrix of m//f X n//f dimension
	for i in range(0, m):
		for j in range(0, n):
			try:
				down_img[i, j] = input_img[i*factor,j*factor]                  # Iterate all elements and multiply index value with factor to downsample
			except IndexError:
				pass
	return down_img

############################################################# Driver Code ################################################################
if __name__=='__main__':
	# Read the image
	image = cv2.imread('Images\city.png', 0) 
	m, n = image.shape

    # Plot the original image
	plt.subplot(2, 2, 1)
	plt.imshow(image, cmap='gray')
	plt.title('Original image')
    
	# Plot downsampled image
	plt.subplot(2, 2, 2)
	img1 = down_sample(image, 2)
	plt.imshow(img1, cmap='gray')
	plt.title('Downsampling factor 2')

	plt.subplot(2, 2, 3)
	img2 = down_sample(image, 4)
	plt.imshow(img2, cmap='gray')
	plt.title('Downsampling factor 4')

	plt.subplot(2, 2, 4)
	img3 = down_sample(image, 5)
	plt.imshow(img3, cmap='gray')
	plt.title('Downsampling factor 5')
	plt.show()