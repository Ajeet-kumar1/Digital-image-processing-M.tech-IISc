# Import pre-requisites
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define downsampling function
def down_sample(input_img, m, n, factor):
	
	down_img = np.zeros((m//factor, n//factor))                                # Create an output matrix of m//f X n//f dimension
	for i in range(0, m):
		for j in range(0, n):
			try:
				down_img[i, j] = input_img[i*factor,j*factor]                  # Iterate all elements and multiply index value with factor to downsample
			except IndexError:
				pass
	return down_img


# Define upsampling with nearest neighbour
def up_sample_nearest_neigh(down_img, m, n, factor):
	up_samp_neigh = np.zeros((m,n))                                           # Define a matrix having original size
	for i in range(0, m):
		for j in range(0, n):
			try:
				up_samp_neigh[i, j] = down_img[round(i/factor), round(j/factor)] # Find the element to nearest neighbour by round off
			except IndexError:
				pass
	
	return up_samp_neigh



# Define Interpolation function ##########		
def up_sample_inter(img2):
	factor = 3
	m, n = img2.shape
	output = np.zeros((m, n))                                     # # Define an output image of zeros
	I = img2                                                      # # To make expression easy
	for i in range(0, m):
		for j in range(0, n):
			t = i/factor                                # find the upsampled index
			r = j/factor
			t1 = round(np.floor(t))
			r1 = round(np.floor(r))
			A = [[1,0,0,0],[1,1,0,0],[1, 0, 1,0],[1,1,1,1]]       # Define coefficient matrix to solve linear equation
			Y = [I[t1,r1], I[t1+1, r1], I[t1,r1+1], I[t1+1,r1+1]] # Write the constant value matrix
			res = np.linalg.inv(A).dot(Y)                         # Solve it using linear algebra
			output[i, j] = round(res[0] + res[1]*(t - t1) + res[2]*(r -r1) +res[3]*(r-r1)*(t -t1))  # Get the output value by putting in formula
	return output






if __name__=='__main__':
	# Read flowers.png and convert into gray scale
	flower = cv2.imread('images\Flowers.png')
	flower_gray  = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
	m, n = flower_gray.shape
	factor = 3
	#  Call the downsample function
	down_img = down_sample(flower_gray, m, n, factor)
	#Plot the original and downsampled image
	plt.subplot(2,2,1)
	plt.imshow(flower_gray, cmap='gray')
	plt.title('Original image')
	plt.subplot(2,2,2)
	plt.imshow(down_img, cmap="gray")
	plt.title('Down sampled image')
	# call upsampled image
	up_sample_near = up_sample_nearest_neigh(down_img, m, n, factor)
	#up_sample_int = dummy(down_img)
	#Plot the up sampled image
	plt.subplot(2,2,3)
	plt.imshow(up_sample_near, cmap='gray')
	plt.title('Up sampled with nearest')
	#plt.figure(2)
	plt.subplot(2,2,4)
	up_sample_int = up_sample_inter(down_img)
	plt.imshow(up_sample_int, cmap='gray')
	plt.title('Up sampled with interpolation')
	plt.show()




##############################

