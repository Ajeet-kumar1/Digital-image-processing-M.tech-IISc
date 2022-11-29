from downsample import*
from filtered_downsampling import*
from image_smoothing import*
from edge_detection_prewitt import*
from edge_rotated_image import*
from edge_detection_laplace import*
from harris_corner import*
from noisy_image_corner import*
from harris_synthetic_image import*







############################################ Driver code for all #######################################
i = input("Press Enter in terminal through keyboard to run the program 1 (a): ")
print('Please wait....')

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

#################################################################################################################
i = input("Press Enter in terminal through keyboard to run the program 1 (b): ")
print('Please wait....')

# Read the image
image = cv2.imread('Images\city.png', 0)
m, n = image.shape

# Perform image smoothing
kernel = gaussiun_kernel(5, 0, 2)
convolved_img = convolution(image, kernel)

# Plot the manually filtered image
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(convolved_img, cmap='gray')
plt.title('Manually Filtered image')

# Plot the libarary output for comparision 
plt.subplot(1,2,2)
blur = cv2.GaussianBlur(image,(5,5),2)
plt.imshow(blur, cmap='gray')
plt.title('Libarary filtered output')

# Apply downsampling on manually filtered image
down_img_2 = down_sample(convolved_img, 2)
down_img_4 = down_sample(convolved_img, 4)
down_img_5 = down_sample(convolved_img, 5)

# Apply downsampling of library filtered image
down_img_2_blur = down_sample(blur, 2)
down_img_4_blur = down_sample(blur, 4)
down_img_5_blur = down_sample(blur, 5)

# Now plot the manually filtered downsampled image
plt.figure(2)
plt.subplot(2, 3, 1)
plt.imshow(down_img_2, cmap='gray')
plt.title('Downsampled by 2 manually filtered image')

# Downsample by factor 4
plt.subplot(2, 3, 2)
plt.imshow(down_img_4, cmap='gray')
plt.title('Downsampled by 4 manually filtered image')

# Downsample by factor 5
plt.subplot(2, 3, 3)
plt.imshow(down_img_5, cmap='gray')
plt.title('Downsampled by 5 manually filtered image')

# Now plot the library filtered downsample image
plt.subplot(2, 3, 4)
plt.imshow(down_img_2_blur, cmap='gray')
plt.title('Downsampled by 2 library filtered image')

plt.subplot(2, 3, 5)
plt.imshow(down_img_4_blur, cmap='gray')
plt.title('Downsampled by 4 library filtered image')

plt.subplot(2, 3, 6)
plt.imshow(down_img_5_blur, cmap='gray')
plt.title('Downsampled by 5 library filtered image')
plt.show()

#############################################################################################################################################
i = input("Press Enter in terminal through keyboard to run the program 2 (a): ")
print('Please wait....')

kernel = gaussiun_kernel(5, 0, 1)                                                  # 5 X 5 kernel with variance 25
    
checkerboard_image = cv2.imread('Images\Checkerboard.png', 0)                      # Read all images   
noisy_checker_image = cv2.imread('Images\oisyCheckerboard.png', 0)
coin_image = cv2.imread('Images\Coins.png', 0)
noisy_coin = cv2.imread('Images\oisyCoins.png', 0)

####################################### Apply filtering on them #######
checkerboard_out = smoothing(checkerboard_image, kernel)
blur1 = cv2.GaussianBlur(checkerboard_image,(5,5),0)                               # Use library function for comparison

noisy_checker_out = smoothing(noisy_checker_image, kernel)
blur2 = cv2.GaussianBlur(noisy_checker_image,(5,5),0)

coin_out = smoothing(coin_image, kernel)
blur3 = cv2.GaussianBlur(coin_image,(5,5),0)

noisy_coin_out = smoothing(noisy_coin, kernel)
blur4 = cv2.GaussianBlur(noisy_coin,(5,5),0)

############################################################# Plot them #####

plt.subplot(2, 4, 1)
plt.imshow(checkerboard_out, cmap='gray')
plt.title('Checker board output')

plt.subplot(2, 4, 2)
plt.imshow(noisy_checker_out, cmap='gray')
plt.title('Noisy board output')

plt.subplot(2, 4, 3)
plt.imshow(coin_out, cmap='gray')
plt.title('Coin image output')

plt.subplot(2, 4, 4)
plt.imshow(noisy_coin_out, cmap='gray')
plt.title('Noisy coin output')

####################################################### Plot the libarary function output for comparison ##########################
plt.subplot(2, 4, 5)
plt.imshow(blur1, cmap='gray')
plt.title('Checker board library output')

plt.subplot(2, 4, 6)
plt.imshow(blur2, cmap='gray')
plt.title('Noisy checker libarary output')

plt.subplot(2, 4, 7)
plt.imshow(blur3, cmap='gray')
plt.title('Coin image libaray output')

plt.subplot(2, 4, 8)
plt.imshow(blur4, cmap='gray')
plt.title('Noisy coin library output')
plt.show()

#############################################################################################################################################
i = input("Press Enter in terminal through keyboard to run the program 2 (b): ")
print('Please wait....')
kernel = gaussiun_kernel(5, 0, 1)                                                                   # Evaluate the gaussian kernel
kernel_x = np.array([                                                                               # Define x-axis and y-axis differentiator
                    [-1,-1,-1],
                    [0, 0, 0],
                    [1, 1, 1]
                            ])

kernel_y = np.array([
                    [-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]
                            ])

checkerboard_image = cv2.imread('Images\Checkerboard.png', 0)                      # Read all images   
noisy_checker_image = cv2.imread('Images\oisyCheckerboard.png', 0)
coin_image = cv2.imread('Images\Coins.png', 0)
noisy_coin = cv2.imread('Images\oisyCoins.png', 0)

#################################################################### Plot the original image ############################################################

plt.subplot(2, 4, 1)
plt.imshow(checkerboard_image, cmap='gray')
plt.title('Original image')

plt.subplot(2, 4, 2)
plt.imshow(noisy_checker_image, cmap='gray')
plt.title('Original image')

plt.subplot(2, 4, 3)
plt.imshow(coin_image, cmap='gray')
plt.title('Original image')

plt.subplot(2, 4, 4)
plt.imshow(noisy_coin, cmap='gray')
plt.title('Original image')

################################################################### Now plot the edges of image #########################################################

edge_checkerboard = edge_detect(checkerboard_image, kernel_x, kernel_y)
plt.subplot(2, 4, 5)
plt.imshow(edge_checkerboard, cmap='gray')
plt.title('Edges of image without filtering')

edge_noisy_checker = edge_detect(noisy_checker_image, kernel_x, kernel_y)
plt.subplot(2, 4, 6)
plt.imshow(edge_noisy_checker, cmap='gray')
plt.title('Edges of image without filtering')

edge_coin = edge_detect(coin_image, kernel_x, kernel_y)
plt.subplot(2, 4, 7)
plt.imshow(edge_coin, cmap='gray')
plt.title('Edges of image without filtering')

edge_noisy = edge_detect(noisy_coin, kernel_x, kernel_y)
plt.subplot(2, 4, 8)
plt.imshow(edge_noisy, cmap='gray')
plt.title('Edges of image without filtering')

#################################################################Apply gaussian Smoothing###### ######################################################
checkerboard_smooth = smoothing(checkerboard_image, kernel)
noisy_checker_smooth = smoothing(noisy_checker_image, kernel ) 
coin_smooth = smoothing(coin_image,  kernel)
noisy_coin_smooth = smoothing(noisy_coin, kernel)

################################################################## Plot the smoothen image ###############################################################
plt.figure(2)
plt.subplot(2, 4, 1)
plt.imshow(checkerboard_smooth, cmap='gray')
plt.title('Smoothen image')

plt.subplot(2, 4, 2)
plt.imshow(noisy_checker_smooth, cmap='gray')
plt.title('Smoothen image')

plt.subplot(2, 4, 3)
plt.imshow(coin_smooth, cmap='gray')
plt.title('Smoothen image')

plt.subplot(2, 4, 4)
plt.imshow(noisy_coin_smooth, cmap='gray')
plt.title('Smoothen image')

################################################################### Now plot the edges of smoothen image #########################################################

edge_checkerboard = edge_detect(checkerboard_smooth, kernel_x, kernel_y)
plt.subplot(2, 4, 5)
plt.imshow(edge_checkerboard, cmap='gray')
plt.title('Edges of image with smoothing')

edge_noisy_checker = edge_detect(noisy_checker_smooth, kernel_x, kernel_y)
plt.subplot(2, 4, 6)
plt.imshow(edge_noisy_checker, cmap='gray')
plt.title('Edges of image with smoothing')

edge_coin = edge_detect(coin_smooth, kernel_x, kernel_y)
plt.subplot(2, 4, 7)
plt.imshow(edge_coin, cmap='gray')
plt.title('Edges of image with smoothing')

edge_noisy = edge_detect(noisy_coin_smooth, kernel_x, kernel_y)
plt.subplot(2, 4, 8)
plt.imshow(edge_noisy, cmap='gray')
plt.title('Edges of image with smoothing')
plt.show()

#############################################################################################################################################
i = input("Press Enter in terminal through keyboard to run the program 2 (c): ")
print('Please wait....')

kernel = edge_detection_prewitt.gaussiun_kernel(5, 0, 1)
kernel_x = np.array([
                    [-1,-1,-1],
                    [0, 0, 0],
                    [1, 1, 1]
                            ])

kernel_y = np.array([
                    [-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]
                            ])

checkerboard_image = cv2.imread('Images\Checkerboard.png', 0)                      # Read all images   
noisy_checker_image = cv2.imread('Images\oisyCheckerboard.png', 0)
coin_image = cv2.imread('Images\Coins.png', 0)
noisy_coin = cv2.imread('Images\oisyCoins.png', 0)

##################################################################### Rotate the image ##################################################################
checkerboard_image = cv2.rotate(checkerboard_image, cv2.ROTATE_90_CLOCKWISE)
noisy_checker_image = cv2.rotate(noisy_checker_image, cv2.ROTATE_90_CLOCKWISE)
coin_image = cv2.rotate(coin_image, cv2.ROTATE_90_CLOCKWISE)
noisy_coin = cv2.rotate(noisy_coin, cv2.ROTATE_90_CLOCKWISE)


#################################################################### Plot the original image ############################################################

plt.subplot(2, 4, 1)
plt.imshow(checkerboard_image, cmap='gray')
plt.title('Rotated image')

plt.subplot(2, 4, 2)
plt.imshow(noisy_checker_image, cmap='gray')

plt.subplot(2, 4, 3)
plt.imshow(coin_image, cmap='gray')

plt.subplot(2, 4, 4)
plt.imshow(noisy_coin, cmap='gray')

################################################################### Now plot the edges of image #########################################################

edge_checkerboard = edge_detect(checkerboard_image, kernel_x, kernel_y)
plt.subplot(2, 4, 5)
plt.imshow(edge_checkerboard, cmap='gray')
plt.title('Edges of image after rotation')

edge_noisy_checker = edge_detect(noisy_checker_image, kernel_x, kernel_y)
plt.subplot(2, 4, 6)
plt.imshow(edge_noisy_checker, cmap='gray')

edge_coin = edge_detect(coin_image, kernel_x, kernel_y)
plt.subplot(2, 4, 7)
plt.imshow(edge_coin, cmap='gray')

edge_noisy = edge_detect(noisy_coin, kernel_x, kernel_y)
plt.subplot(2, 4, 8)
plt.imshow(edge_noisy, cmap='gray')
plt.show()

#############################################################################################################################################
i = input("Press Enter in terminal through keyboard to run the program 2 (d): ")
print('Please wait....')


# Define laplacian operator
laplace_kernel = np.array([
                    [0,-1, 0],
                    [-1,4,-1],
                    [0, -1,0]
                            ])

checkerboard_image = cv2.imread('Images\Checkerboard.png', 0)                      # Read all images   
noisy_checker_image = cv2.imread('Images\oisyCheckerboard.png', 0)
coin_image = cv2.imread('Images\Coins.png', 0)
noisy_coin = cv2.imread('Images\oisyCoins.png', 0)

#################################################################### Plot the original image ############################################################

plt.subplot(2, 4, 1)
plt.imshow(checkerboard_image, cmap='gray')
plt.title('Original image')

plt.subplot(2, 4, 2)
plt.imshow(noisy_checker_image, cmap='gray')
plt.title('Original image')

plt.subplot(2, 4, 3)
plt.imshow(coin_image, cmap='gray')
plt.title('Original image')

plt.subplot(2, 4, 4)
plt.imshow(noisy_coin, cmap='gray')
plt.title('Original image')

################################################################### Now plot the edges of image ####################################################

edge_checkerboard = edge_dection(checkerboard_image, laplace_kernel)
plt.subplot(2, 4, 5)
plt.imshow(edge_checkerboard, cmap='gray')
plt.title('Edges of image')

noisy_checker_smooth = smoothing(noisy_checker_image, gaussiun_kernel(5, 0, 1) )                           # For noisy first smooth
edge_noisy_checker = edge_dection(noisy_checker_smooth, laplace_kernel)
edge_noisy_checker = (edge_noisy_checker > 5) * edge_noisy_checker                                       # Threshold in case on noisy image
plt.subplot(2, 4, 6)
plt.imshow(edge_noisy_checker, cmap='gray')
plt.title('Edges of image after thresold= 5')

edge_coin = edge_dection(coin_image, laplace_kernel)
plt.subplot(2, 4, 7)
plt.imshow(edge_coin, cmap='gray')
plt.title('Edges of image')

noisy_coin = smoothing(noisy_coin, gaussiun_kernel(5, 0, 1) )
edge_noisy = edge_dection(noisy_coin, laplace_kernel)
plt.subplot(2, 4, 8)
edge_noisy = (edge_noisy > 15) * edge_noisy                                                              # Threshold noisy image 
plt.imshow(edge_noisy, cmap='gray')
plt.title('Edges of image after thresold = 15')
plt.show()

#############################################################################################################################################
i = input("Press Enter in terminal through keyboard to run the program 3 (a): ")
print('Please wait....')

checkerboard = cv2.imread('Images\Checkerboard.png', 0)                                 # Read the both test images
mainbuilding = cv2.imread('Images\MainBuilding.png', 0)

img_corners_checker = harris_corner(image=checkerboard, thresold=10,sensitivity=0.05,sigma=2) # Call the function to detect the corner
img_corners_mainb = harris_corner(image=mainbuilding,thresold=50,sensitivity=0.05,sigma=0.2) 

plt.subplot(2, 2, 1)                                                                    # Plot the original image
plt.imshow(checkerboard, cmap='gray')
plt.title('Original image')
        
plt.subplot(2, 2, 2)                                                                    # Plot the corner of image
plt.imshow(img_corners_checker, cmap='gray')
plt.title('Corner of checker board at thresold 0')

plt.subplot(2, 2, 3)                                                                    # Plot the original image of main building
plt.imshow(mainbuilding, cmap='gray')
plt.title('Original image of main building')

plt.subplot(2, 2, 4)                                                                    # Plot the corner of image
plt.imshow(img_corners_mainb, cmap='gray')
plt.title('Corner of main building at thresold 0')
plt.show()


#############################################################################################################################################
i = input("Press Enter in terminal through keyboard to run the program 3 (b): ")
print('Please wait....')

# Read the both test images
checkerboard = cv2.imread('Images\Checkerboard.png', 0)                                           
mainbuilding = cv2.imread('Images\MainBuilding.png', 0)

# Apply rotate scale and noise addition on image                                                 
n_checkerboard = rotate_scale_noise(checkerboard)
n_mainbuilding = rotate_scale_noise(mainbuilding)


# Detect the corner in modified image
corners_checker = harris_corner(image=n_checkerboard, thresold=10,sensitivity=0.05,sigma=2)
corners_mainb = harris_corner(image=n_mainbuilding,thresold=50,sensitivity=0.05,sigma=0.2)

# Plot the checkerboard original image 
plt.subplot(2, 3, 1)
plt.imshow(checkerboard, cmap='gray')
plt.title('original image')

# Plot noisy image
plt.subplot(2, 3, 2)
plt.imshow(n_checkerboard, cmap='gray')
plt.title('Noisy roted scaled image')

# Plot Corners of noisy rotated image
plt.subplot(2, 3, 3)
plt.imshow(corners_checker, cmap='gray')
plt.title('Corner of Noisy roted scaled image')

# Plot mainbuilding image
plt.subplot(2, 3, 4)
plt.imshow(mainbuilding, cmap='gray')
plt.title('original image')

plt.subplot(2, 3, 5)
plt.imshow(n_mainbuilding, cmap='gray')
plt.title('Noisy roted scaled image')

plt.subplot(2, 3, 6)
plt.imshow(corners_mainb, cmap='gray')
plt.title('Corner of Noisy roted scaled image')
plt.show()


#############################################################################################################################################
i = input("Press Enter in terminal through keyboard to run the program 3 (c): ")
print('Please wait....')

synthetic_image1 = synthetic_image()
synthetic_gray = cv2.cvtColor(synthetic_image1, cv2.COLOR_RGB2GRAY)
corners = harris_corner(image=synthetic_gray,thresold=10,sensitivity=0.05,sigma=1)

plt.subplot(1, 2, 1)
plt.imshow(synthetic_image1, cmap='gray')
plt.title('Original synthetic image')

plt.subplot(1, 2, 2)
plt.imshow(corners, cmap='gray')
plt.title('Corners of image')
plt.show()
print('Thank you')






