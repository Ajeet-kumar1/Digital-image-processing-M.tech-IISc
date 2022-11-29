# Import required packages and functions
import numpy as np
import cv2
import matplotlib.pyplot as plt
from full_scale_contrast_stretching import streching

# Define Equalization function
def equalizer(img1):
   
   # In order to get good output first perform contrast streching
   _, _, img1, k, Hk = streching(img1)
   row, col = img1.shape
   Mn = row * col
   
   # Find the cumulative sum
   Pk = np.divide(Hk, Mn)
   Pk = np.cumsum(Pk)

   # Convert pixel value array and cumulative array into list
   Pk1 =list(Pk)
   k = list(k)
   # Hard code for cumulative( I have not used it to reduce complexity)

   #for i in range(256):
      #if i==0:
        # Pk1.append(Pk[0])
      #else:
       #  Pk1.append(Pk[i]+Pk1[i-1])
 
   # Apply equalization on each pixel
   for i in range(row):
      for j in range(col):
         img1[i,j] = ((Pk1[k.index(img1[i,j])]-Pk1[1])/(1-Pk1[1])) *  255 
   return img1

if __name__== '__main__':
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



