import cv2
import matplotlib.pyplot as plt
import numpy as np
from full_scale_contrast_stretching import streching
from histogram_equalization import equalizer
from clahe import clahe_no_overlap, clahe_with_overlap
from sampling import down_sample, up_sample_nearest_neigh, up_sample_inter
from spatial_filtering import convolution, add_padding


a = int(input(" To run the question 1(a) Enter '1'.\n To run the question 1(b) Enter '2'.\n To run the question 1(c) Enter '3'.\n To run the question 2 Enter '4'. \n To run the question 3 Enter '5'. \n"))


if a==1:
    print('Please wait for 8 or more seconds')
    # Read the ECE.png image and convert into gray scale
    img_ECE = cv2.imread('images\ECE.png')
    img11=img_ECE
    img_ECE_gray = cv2.cvtColor(img_ECE, cv2.COLOR_BGR2GRAY)

    # Read the IIScMain.png and convert into gray scale
    img_IISc = cv2.imread('images\IIScMain.png')
    img_IISc_gray = cv2.cvtColor(img_IISc, cv2.COLOR_BGR2GRAY)
    
    # Find histogram and do the streching of image and plot original and streched histogram
    x,y,J, x1,y1 = streching(img_ECE_gray)
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.bar(x,y)
    plt.title('Original bar graph of ECE.png')
    plt.subplot(2,2,2)
    plt.bar(x1,y1)
    plt.title('Streched of ECE.png')
   
   # Also show the original and streched picture
    plt.figure(2)
    plt.subplot(2,2,1)
    plt.imshow(img11,cmap='gray',vmin=0,vmax=255)
    plt.title('original of ECE.png')
    plt.subplot(2,2,2)
    plt.imshow(J,cmap='gray',vmin=0,vmax=255)
    plt.title('Stretched of ECE.png')
    
    # Now do the streching on IIScMain image also and plot both histograms
    x,y,J, x1,y1 = streching(img_IISc_gray)
    plt.figure(1)
    plt.subplot(2,2,3)
    plt.bar(x,y)
    plt.title('Original of IIScMain')
    plt.subplot(2,2,4)
    plt.bar(x1,y1)
    plt.title('Stretched of IIScMain')
   
   # Display the original and streched image
    plt.figure(2)
    plt.subplot(2,2,3)
    plt.imshow(img_IISc_gray,cmap='gray',vmin=0,vmax=255)
    plt.title('Original of IIScMain')
    plt.subplot(2,2,4)
    plt.imshow(J,cmap='gray',vmin=0,vmax=255)
    plt.title('Stretched of IIScMain')
    plt.show()
   
######################################################################## Q 1(b)
elif a==2:
    print('Please wait for 15 or more seconds')
    # Read the lion.png, Hazy.png, and StoneFace.png and convert them into grayscale
    img11 = cv2.imread('images\lion.png')
    img1 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
    img22 = cv2.imread('images\Hazy.png')
    img2 = cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
    img33 = cv2.imread('images\StoneFace.png')
    img3 = cv2.cvtColor(img33, cv2.COLOR_BGR2GRAY)
   
   # Evaluate the equalization of each image
    out1= equalizer(img1)
    out2= equalizer(img2)
    out3= equalizer(img3)
    # Now plot the graph of each input and output image
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(img11,cmap='gray',vmin=0,vmax=255)
    plt.title('Lion original image')
    plt.subplot(1,2,2)
    plt.imshow(out1,cmap='gray',vmin=0,vmax=255)
    plt.title('Lion contrast streched and equalized image')


    plt.figure(2)
    plt.subplot(1,2,1)
    plt.imshow(img22,cmap='gray',vmin=0,vmax=255)
    plt.title('Original of Hazy image')
    plt.subplot(1,2,2)
    plt.imshow(out2,cmap='gray',vmin=0,vmax=255)
    plt.title('Hazy contrast strechecd and equalized image')

    plt.figure(3)

    plt.subplot(1,2,1)
    plt.imshow(img33,cmap='gray',vmin=0,vmax=255)
    plt.title('StoneFace original image')
    plt.subplot(1,2,2)
    plt.imshow(out3,cmap='gray',vmin=0,vmax=255)
    plt.title('StoneFace contrast strechecd and equalized image')
    plt.show()


#######################################################################Q1(c)

elif a==3:
    print('Please wait for 12 or more seconds')
    img_lion = cv2.imread('images\lion.png')
    gray_lion = cv2.cvtColor(img_lion, cv2.COLOR_BGR2GRAY)
    lion_ahe = clahe_no_overlap(gray_lion)
    img_stone = cv2.imread('images\StoneFace.png')
    gray_stone = cv2.cvtColor(img_stone, cv2.COLOR_BGR2GRAY)
    stone_ahe = clahe_no_overlap(gray_stone)
    
    plt.figure(1)
    plt.imshow(lion_ahe, cmap='gray', vmin=0, vmax=255)
    plt.title('Lion adapative equalization')
        
    plt.figure(2)
    plt.imshow(stone_ahe, cmap='gray', vmin=0, vmax=255)
    plt.title('StoneFace adapative equalization')

    stone_clahe = clahe_with_overlap(gray_stone)
    lion_clahe = clahe_with_overlap(gray_lion)
    
    plt.figure(3)
    plt.imshow(lion_clahe, cmap='gray', vmin=0, vmax=255)
    plt.title('Lion contrast limiting adapative equalization with overlap')
        
    plt.figure(4)
    plt.imshow(stone_clahe, cmap='gray', vmin=0, vmax=255)
    plt.title('StoneFace contrast limit apative equalization with overlap')
    plt.show()

######################################################################Q2
elif a==4:
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


###################################################Q 3
if a==5:
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
    print('Please wait few seconds')
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

else:
    print('Please enter a valid input')


